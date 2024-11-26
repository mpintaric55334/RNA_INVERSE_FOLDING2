import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder, PositionalEncoding
from embedding import Embedder


class RNAModel(nn.Module):

    def __init__(self):
        """
        First class
        """
        super(RNAModel, self).__init__()
        self.embedding = Embedder(embedd_size=4, edge_types=2, bin_size=32,
                                  device="cuda")
        self.encoder = Encoder(num_blocks=1, embedd_size=4,
                               num_heads=1, conv_out_channels=50,
                               kernel_size_conv=3,
                               reduction=1, attention_dropout=0,
                               bias=False)
        self.decoder = Decoder(num_blocks=1, embedd_size=4, num_heads=1,
                               dropout=0,
                               attention_dropout=0, attention_bias=False,
                               fc_bias=False, fc_dropout=0)
        self.pos_encoding = PositionalEncoding(embedd_size=4, max_len=200,
                                               device="cuda")

    def forward(self, matrix, seq, attn_mask_enc=None, attn_mask_dec=None):
        matrix = self.embedding(matrix)
        enc = self.encoder(matrix, attn_mask_enc)
        #  seq needs to be transformed using nn.embedding, will add
        seq = self.pos_encoding(seq)
        out = self.decoder(seq, enc, attn_mask_dec, attn_mask_enc)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix = torch.randint(0, 1, (1, 4, 4)).to(device)
seq = torch.randn((1, 4, 4)).to(device)
attn_mask_dec = torch.tril(torch.ones(4, 4, dtype=torch.bool),
                           diagonal=0).to(device)
attn_mask_dec[-1, :] = False
attn_mask_enc = torch.ones(4, 4, dtype=torch.bool).to(device)
attn_mask_enc[-1, :] = False
attn_mask_enc[:, -1] = False
attn_mask_enc = attn_mask_enc.unsqueeze(0)
model = RNAModel().to(device)
out = model(matrix, seq, attn_mask_enc, attn_mask_dec)