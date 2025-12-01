__all__ = ['AHuberBackbone']

from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN

class AHuberBackbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,
                 e_layers:int=3, d_model=128, n_heads=16,
                 d_ff:int=256, norm:str='LayerNorm', attn_dropout:float=0.2, dropout:float=0.3, act:str="gelu",
                 res_attention:bool=True, pre_norm:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0.3, padding_patch = "end",
                 revin = True, affine = True, subtract_last = False,
                 **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1

        self.n_vars = c_in
       
        self.embedding = nn.Linear(patch_len, d_model)
        
        # Create variable (spatial) dimension positional encoding
        self.pos_var = nn.Parameter(torch.randn(1, self.n_vars, 1, d_model))
        
        # Create Patch (temporal) dimension positional encoding
        self.pos_time = nn.Parameter(torch.randn(1, 1, patch_num, d_model))

        self.dropout = nn.Dropout(dropout)

        # Use the unique EchoLayer
        self.perception_layers = nn.ModuleList([
            EchoLayer( 
                d_model, n_heads, d_ff=d_ff, dropout=dropout,
                activation=act, norm=norm, attn_dropout=attn_dropout,
                res_attention=res_attention, pre_norm=pre_norm
            ) for _ in range(e_layers)
        ])

        # Calculate flattened feature dimension
        nf = patch_num * d_model
        
        # Flatten head
        self.flatten_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(head_dropout),
            nn.Linear(nf, target_window)
        )


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z_for_graph = z.permute(0, 2, 1)
            z_norm = self.revin_layer(z_for_graph, 'norm')
            z = z_norm.permute(0,2,1)
        else:
            z_norm = z.permute(0, 2, 1)

    
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        raw_patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)         # raw_patches: [bs x nvars x patch_num x patch_len]

        # Input encoding
        z = self.embedding(raw_patches)                                                     # z: [bs x nvars x patch_num x d_model]
        
        # Apply 2D positional encoding
        z = z + self.pos_var + self.pos_time
        z = self.dropout(z)                                                                 # z: [bs x nvars x patch_num x d_model]

        # Holistic Perception Backbone
        # Initialize hub_contexts for cross-layer transmission
        prev_hub_contexts = None
        
        for i, layer in enumerate(self.perception_layers):
            # Pass prev_hub_contexts to each layer
            z, current_hub_contexts = layer(z, prev_hub_contexts=prev_hub_contexts)
            prev_hub_contexts = current_hub_contexts

        z = self.flatten_head(z) 

        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)

        return z


class EchoLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1, activation='gelu',
                 norm='LayerNorm', attn_dropout=0.1, res_attention=True, pre_norm=False):
        super().__init__()
        
        self.n_heads = n_heads 
        
        # Stage 1: Hub-Aggregation Attention
        self.hub_aggregation_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention, use_hubs=True)
        self.norm1 = get_norm_layer(norm, d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Stage 2: Hub-Distribution Attention
        self.dist_query_proj = nn.Linear(d_model, d_model) # Patch -> Query
        self.dist_key_proj = nn.Linear(d_model, d_model)   # Hub -> Key
        self.dist_value_proj = nn.Linear(d_model, d_model) # Hub -> Value
        self.dist_out_proj = nn.Linear(d_model, d_model)   # Output projection for mixing multi-head info
        
        self.norm2 = get_norm_layer(norm, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Stage 3: FFN
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.norm3 = get_norm_layer(norm, d_model)
        self.dropout3 = nn.Dropout(dropout)
        
        self.pre_norm = pre_norm
        
    def forward(self, x: Tensor, prev_hub_contexts: Optional[Tensor] = None):
        bs, n_vars, patch_num, d_model = x.shape
        x_flat = x.reshape(bs, n_vars * patch_num, d_model)

        # 1. ========== Aggregation Stage (Attention to Hubs) ==========
        x_norm1 = self.norm1(x_flat) if self.pre_norm else x_flat
        hub_contexts, attn_weights, _ = self.hub_aggregation_attn(Q=None, K=x_norm1, V=x_norm1, prev_hub_contexts=prev_hub_contexts)
        
        # 2. ========== Distribution Stage (Distribution via Attention) ==========
        x_res1 = x_flat
        x_norm2 = self.norm2(x_flat) if self.pre_norm else x_flat
        
        # 1. Prepare Query (from Patch) and Key/Value (from Hub)
        # Introduce multi-head mechanism: [bs, Seq, d] -> [bs, Seq, n_heads, d_head] -> [bs, n_heads, Seq, d_head]
        d_head = d_model // self.n_heads
        
        Q_patch = self.dist_query_proj(x_norm2).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        K_hub = self.dist_key_proj(hub_contexts).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        V_hub = self.dist_value_proj(hub_contexts).view(bs, -1, self.n_heads, d_head).transpose(1, 2)
        
        # 2. Calculate Attention Score (Simplified Cross Attention)
        # Q * K^T
        # [bs, n_heads, N*P, d_head] * [bs, n_heads, d_head, 1] -> [bs, n_heads, N*P, 1]
        attn_score = torch.matmul(Q_patch, K_hub.transpose(-1, -2))
        attn_score = attn_score / (d_head ** 0.5) # Note: scaling factor changed to d_head
        attn_weights = torch.sigmoid(attn_score) # Keep Sigmoid gating (since Key length is 1, Softmax is meaningless)
        
        # 3. Weighted Sum (Broadcast V_hub and multiply by gate)
        # [bs, n_heads, N*P, 1] * [bs, n_heads, 1, d_head] -> [bs, n_heads, N*P, d_head]
        updates = attn_weights * V_hub 
        
        # 4. Restore shape and fuse multi-heads
        # [bs, n_heads, N*P, d_head] -> [bs, N*P, n_heads, d_head] -> [bs, N*P, d_model]
        updates = updates.transpose(1, 2).reshape(bs, -1, d_model)
        updates = self.dist_out_proj(updates) # Linear mixing of different heads
        
        x_flat = x_res1 + self.dropout2(updates)
        if not self.pre_norm: x_flat = self.norm2(x_flat)
        
        # 3. ========== FFN Stage ==========
        x_res2 = x_flat
        x_norm3 = self.norm3(x_flat) if self.pre_norm else x_flat
        ff_out = self.ff(x_norm3)
        
        x_flat = x_res2 + self.dropout3(ff_out)
        if not self.pre_norm: x_flat = self.norm3(x_flat)

        output = x_flat.reshape(bs, n_vars, patch_num, d_model)

        return output, hub_contexts


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False, use_hubs=False):
        """Multi Head Attention Layer - Now optimized for Hub-Aggregation ONLY"""
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.use_hubs = use_hubs
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        # === Core Change: Introduce Hub Mode ===
        if self.use_hubs:
            # When acting as a hub, Q is an internal learnable parameter representing independent queries for each head
            # Shape: [1, n_heads, 1, d_k], representing 1 batch, n heads, 1 query per head, d_k dim per query
            self.hub_queries = nn.Parameter(torch.randn(1, self.n_heads, 1, self.d_k))
            self.W_Q = None # In hub mode, no projection for external Q needed
        else:
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None,
                # --- Add prev_hub_contexts parameter ---
                prev_hub_contexts:Optional[Tensor]=None):
        bs = K.size(0)

        # Use internal learnable hub as query
        if self.use_hubs:
            # Expand to current batch size
            q_s = self.hub_queries.expand(bs, -1, -1, -1)
        else:
            # Traditional mode: project external Q
            q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)

        # --- Core Change: Inject previous layer's macro state into current layer's query ---
        if prev_hub_contexts is not None:
            # Logic handled after sdp_attn for simplicity in this implementation
            pass 

        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)

        output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        # --- Core Change: Fuse previous layer context here ---
        if prev_hub_contexts is not None:
            # Assuming d_model == n_heads * d_v, we can add directly
            output = output + prev_hub_contexts
        

        return output, attn_weights, attn_scores

class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            attn_mask       : [bs x 1 x q_len x seq_len] or [bs x n_heads x q_len x seq_len]
        '''
        attn_scores = torch.matmul(q, k) * self.scale

        if prev is not None: attn_scores = attn_scores + prev

        if attn_mask is not None:
            # Allows graph structure to influence attention flexibly
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights, attn_scores
    
def get_norm_layer(norm_type, d_model):
    if "batch" in norm_type.lower():
        return nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
    else:
        return nn.LayerNorm(d_model)
    
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
        
def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')
