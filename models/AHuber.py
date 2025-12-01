# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.AHuber_backbone import AHuberBackbone


class Model(nn.Module):
    def __init__(self, configs, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='LayerNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, head_type = 'flatten', **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        e_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        head_dropout = configs.head_dropout
        stride = configs.stride
        patch_len = configs.patch_len
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        # model
        
        self.model = AHuberBackbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                    e_layers=e_layers, d_model=d_model,
                    n_heads=n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                    dropout=dropout, act=act, res_attention=res_attention, pre_norm=pre_norm,
                    pe=pe, learn_pe=learn_pe, head_dropout=head_dropout,
                    revin=revin, affine=affine, subtract_last=subtract_last)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
       
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]

        z = self.model(x)
        z = z.permute(0, 2, 1)    # z: [Batch, Input length, Channel]
        return z