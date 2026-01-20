import torch
import torch.nn as nn
from layers.AHuber_backbone import AHuberBackbone

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.model = AHuberBackbone(
            c_in=configs.enc_in,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            n_heads=configs.n_heads,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            revin=configs.revin
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, N_Vars]
        
        # Permute for Backbone: [Batch, N_Vars, Seq_Len]
        x = x.permute(0, 2, 1)
        
        # Backbone Forward
        out = self.model(x)
        
        # Permute back: [Batch, Pred_Len, N_Vars]
        out = out.permute(0, 2, 1)
        
        return out