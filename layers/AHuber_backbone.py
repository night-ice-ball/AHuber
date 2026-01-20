import torch
import torch.nn as nn
from layers.RevIN import RevIN

class AHuberBackbone(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, d_model=128, d_ff=256, n_heads=4, e_layers=3, 
                 d_hub=1, dropout=0.1, revin=True):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.revin = revin
        self.d_hub = d_hub
        # 1. RevIN
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        
        # 2. Backbone Layers
        self.hub_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.layers = nn.ModuleList([
            Backbone(seq_len, d_model, d_ff, n_heads, d_hub, dropout)
            for _ in range(e_layers)
        ])

        # 3. Prediction Head
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, N_vars, Seq_len]
        
        # === 1. RevIN Normalization ===
        if self.revin:
            x = x.transpose(1, 2) 
            x = self.revin_layer(x, 'norm')
            x = x.transpose(1, 2)
        bs = x.shape[0]

        current_hub = self.hub_token.expand(bs, -1, -1) # [bs, 1, d_model]
        
        
        for layer in self.layers:

            x, current_hub = layer(x, current_hub)

        # === 3. Prediction ===
        # [B, N, S] -> [B, N, P]
        x_out = self.head(x)
        
        # === 4. RevIN Denormalization ===
        if self.revin:
            x_out = x_out.transpose(1, 2)
            x_out = self.revin_layer(x_out, 'denorm')
            x_out = x_out.transpose(1, 2)
            
        return x_out
    
class Backbone(nn.Module):
    def __init__(self, seq_len, d_model=128, d_ff=256, n_heads=4, d_hub=1, dropout=0.1):
        """
        Time Series -> Token -> Hub Interaction -> Token -> Time Series
        """
        super().__init__()
  
        # 1. Encoder: Seq -> Token

        self.encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. Aggregation: Vars -> Hub
        self.attn_agg = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # 3. Distribution: Hub -> Vars
        self.attn_dist = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # 4. Decoder: Token -> Seq

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len)
        )
        self.norm_hub = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, pre_hub_context):

        # === Step 1: Encoder ===
        var_tokens = self.encoder(x_in)
        
        # === Step 2: Aggregation ===
        # Aggregation: Hub (Query) <- Vars (Key/Value)

        current_hub, _ = self.attn_agg(query=pre_hub_context, key=var_tokens, value=var_tokens)
        
        #  (Hub Residual): next = prev + delta
        next_hub_context= self.norm_hub(pre_hub_context + self.dropout(current_hub))
        
        # === Step 3: Distribution ===
        var_refined, _ = self.attn_dist(query=var_tokens, key=next_hub_context,value=next_hub_context)
        
        dec_in = self.norm_attn(var_tokens + self.dropout(var_refined))
        
        # === Step 4: Decode & Output ===
        dec_out = self.decoder(dec_in)
        # x_out = x_in + dec_out
        x_out = x_in + self.dropout(dec_out)
        
        return x_out, next_hub_context
