import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder, PositionalEncoding
from embedding import EmbedderMatrix, EmbedderSequence


class RNAModel(nn.Module):

    def __init__(self, embedd_size: int, edge_types: int, bin_size: int,
                 device: str, nucl_types: int, num_blocks_enc: int,
                 num_heads: int, conv_out_channels: int,
                 kernel_size_conv: int, reduction: int | tuple[int, int],
                 attention_dropout: float, bias: bool, num_blocks_dec: int,
                 out_dim: int, dropout: float, max_len: int):
        """
        Class of the whole model.
        """
        super(RNAModel, self).__init__()
        """
        Arguments:
            - embedd_size: int => embedding dimension of model
            - edge_types: int => number of possible
            nucleotide token types. Includes embedding
            and unknown tokens
            - device: str => device of the model, serves
            to create intermediate tensors on same device
            - nucl_types: int => number of possible
            nucleotide token types. Includes embedding
            and unknown tokens
            - num_blocks_enc: int => number of encoder blocks
            - num_heads: int => number of attention heads
            - conv_out_channels: int => number of out channels for the middle
            of the convolution layer in encoder
            - kernel_size_conv: int => size of convolution_kernel in conv
            - reduction: int | tuple[int, int] => type of encoder
            output reduction
            - attention_dropout: float => attention dropout
            - bias: bool => should bias be added
            - padding: str => padding for conv in encoder, only same for now
            - num_blocks_dec: int => number of decoder blocks
            - num_heads: int => number of attention heads
            - out_dim: int => number of possible predictions
            - dropout: float => dropout after attentions in decoder
            - max_len: int => max sequence length
        """
        self.embedding_matrix = EmbedderMatrix(embedd_size=embedd_size,
                                               edge_types=edge_types,
                                               bin_size=bin_size,
                                               device=device)
        self.embedding_sequence = EmbedderSequence(embedd_size=embedd_size,
                                                   nucleotide_types=nucl_types)
        self.encoder = Encoder(num_blocks=num_blocks_enc,
                               embedd_size=embedd_size, num_heads=num_heads,
                               conv_out_channels=conv_out_channels,
                               kernel_size_conv=kernel_size_conv,
                               reduction=reduction,
                               attention_dropout=attention_dropout,
                               bias=bias)
        self.decoder = Decoder(num_blocks=num_blocks_dec,
                               embedd_size=embedd_size, num_heads=num_heads,
                               out_dim=out_dim, dropout=dropout,
                               attention_dropout=attention_dropout,
                               attention_bias=bias, fc_bias=bias,
                               fc_dropout=dropout)
        self.pos_encoding = PositionalEncoding(embedd_size=embedd_size,
                                               max_len=max_len,
                                               device=device)

    def forward(self, matrix, seq, attn_mask_enc=None, attn_mask_dec=None):
        matrix = self.embedding_matrix(matrix)
        enc = self.encoder(matrix, attn_mask_enc)
        seq = self.embedding_sequence(seq)
        seq = self.pos_encoding(seq)
        out = self.decoder(seq, enc, attn_mask_dec, attn_mask_enc)
        return out
