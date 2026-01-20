import math
import torch
from torch import nn
from layers.SRS import SRS
from layers.RevIN import RevIN

DEFAULT_HYPER_PARAMS = {
    "hidden_size": 128,
    "d_model": 512,
    "freq": "h",
    "patch_len": 24,
    "stride": 24,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "batch_size": 256,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
    "subtract_last": False,
    "affine": True,
    "head_mode": "linear",
    "alpha": 2.0,
    "pos": True
}

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode='linear'):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == 'linear':
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(), nn.Linear(nf // 2, target_window))
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(Model, self).__init__()
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        
        patch_len = getattr(config, 'patch_len', DEFAULT_HYPER_PARAMS['patch_len'])
        stride = getattr(config, 'stride', DEFAULT_HYPER_PARAMS['stride'])
        d_model = getattr(config, 'd_model', DEFAULT_HYPER_PARAMS['d_model'])
        dropout = getattr(config, 'dropout', DEFAULT_HYPER_PARAMS['dropout'])
        hidden_size = getattr(config, 'hidden_size', DEFAULT_HYPER_PARAMS['hidden_size'])
        alpha = getattr(config, 'alpha', DEFAULT_HYPER_PARAMS['alpha'])
        pos = getattr(config, 'pos', DEFAULT_HYPER_PARAMS['pos'])
        head_dropout = getattr(config, 'head_dropout', DEFAULT_HYPER_PARAMS['head_dropout'])
        head_mode = getattr(config, 'head_mode', DEFAULT_HYPER_PARAMS['head_mode'])
        affine = getattr(config, 'affine', DEFAULT_HYPER_PARAMS['affine'])
        subtract_last = getattr(config, 'subtract_last', DEFAULT_HYPER_PARAMS['subtract_last'])

        self.patch_len = patch_len
        self.stride = stride

        # selective representation space
        self.patch_embedding = SRS(
            d_model, 
            self.patch_len, 
            self.stride, 
            self.seq_len, 
            dropout, 
            hidden_size, 
            alpha, 
            pos
        )

        # Prediction Head
        patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.head_nf = d_model * patch_num
        
        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            self.pred_len,
            head_dropout=head_dropout,
            mode=head_mode
        )

        self.revin = RevIN(num_features=config.enc_in, affine=affine, subtract_last=subtract_last)

    def forward(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out