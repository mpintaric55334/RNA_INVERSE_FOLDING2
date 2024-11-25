import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):

    def __init__(self):
        """
        This is just a concept class for now to see if dummy
        tensor can pass
        """
        super(Model, self).__init__()
        self.encoder = Encoder(num_blocks=1, embedd_size=25,
                               num_heads=5, conv_out_channels=50,
                               kernel_size_conv=3,
                               reduction=1, attention_dropout=0,
                               bias=False)
        self.decoder = Decoder(embedd_size=25, num_heads=5, dropout=0,
                               attention_dropout=0, attention_bias=False,
                               fc_bias=False, fc_dropout=0)

    def forward(self, matrix, seq, attn_mask_enc=None, attn_mask_dec=None):
        enc = self.encoder(matrix, attn_mask_enc)
        out = self.decoder(seq, enc, attn_mask_dec)
        return out


matrix = torch.randn((10, 10, 10, 25))
seq = torch.randn((10, 10, 25))
attn_mask_dec = torch.tril(torch.ones(10, 10, dtype=torch.bool),
                           diagonal=0)
attn_mask_enc = torch.ones(10, 10, dtype=torch.bool)
attn_mask_enc[-1, :] = False
attn_mask_enc[:, -1] = False
attn_mask_enc = attn_mask_enc.unsqueeze(0)
model = Model()
out = model(matrix, seq, attn_mask_enc, attn_mask_dec)