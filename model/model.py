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
        enc = self.encoder(matrix)
        out = self.decoder(seq, enc, attn_mask_dec)
        return enc


matrix = torch.randn((10, 10, 10, 25))
seq = torch.randn((10, 10, 25))
attn_mask_dec = torch.triu(torch.ones(10, 10, dtype=torch.bool),
                           diagonal=1)
attn_mask_enc = torch.zeros(10, 10, dtype=torch.bool)  # there is a problem with this approach to masking, will need to discuss
attn_mask_enc[-1, :] = True
attn_mask_enc[:, -1] = True
model = Model()
out = model(matrix, seq, attn_mask_enc, attn_mask_dec)
print(out.shape)