import torch
import torch.nn as nn
from attention import MultiHeadAttention


class AttentionLayer(nn.Module):
    """
    Class that implements the attention layers of the decoder.
    """

    def __init__(self, embedd_size: int, num_heads: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0, bias: bool = False):
        """
        Arguments:
            - embedd_size: embedd dimension of the model
            - num_heads: number of attention heads
            - dropout: dropout after attentions
            - attention_dropout: dropout of attention
            - bias: will the bias be added to attention
        """
        super(AttentionLayer, self).__init__()

        self.attention = MultiHeadAttention(embedd_size, num_heads,
                                            attention_dropout, bias)
        self.norm = nn.LayerNorm(embedd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        _x = q  # input is always q
        x, _ = self.attention(q, k, v, attn_mask)
        x = self.dropout(x)
        x = self.norm(x + _x)
        return x


class FeedForwardLayer(nn.Module):
    """
    Class that implements the feed forward layer of the decoder.
    """

    def __init__(self, fc_dim: int, bias: bool, dropout: float):
        """
        Arguments:
            - fc_dim: fully connected in and out dimension,
                needs to be same because of the residual connection
            - bias: add bias to fc layer
            - dropout: amount of dropout added to layer
        """
        super(FeedForwardLayer, self).__init__()

        self.linear = nn.Linear(in_features=fc_dim,
                                out_features=fc_dim, bias=bias)
        self.norm = nn.LayerNorm(fc_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _x = x
        x = self.linear(x)
        x = self.dropout(x)
        x = self.norm(x + _x)
        return x


class Decoder(nn.Module):
    """
    Class that implements the decoder.
    """
    def __init__(self, embedd_size: int, num_heads: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 attention_bias: bool = False,
                 fc_bias: bool = False,
                 fc_dropout: float = 0.0):
        """
        Arguments:
            - embedd_size: embedd dimension of the model,
              also the fc_dimension
            - num_heads: number of attention heads
            - dropout: dropout after attentions
            - attention_dropout: dropout of attention
            - attention_bias: will the bias be added to attention
            - fc_bias: add bias to fc layer
            - fc_dropout: amount of dropout added to fc_layer
        """
        super(Decoder, self).__init__()

        self.masked_mh_att = AttentionLayer(embedd_size=embedd_size,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            attention_dropout=attention_dropout,
                                            bias=attention_bias)
        self.cross_att = AttentionLayer(embedd_size=embedd_size,
                                        num_heads=num_heads,
                                        dropout=attention_dropout,
                                        attention_dropout=attention_dropout,
                                        bias=attention_bias)
        self.fc_layer = FeedForwardLayer(fc_dim=embedd_size, bias=fc_bias,
                                         dropout=fc_dropout)

    def forward(self, dec_input, enc_output, attn_mask=None):
        x = self.masked_mh_att(dec_input, dec_input, dec_input,
                               attn_mask)
        x = self.cross_att(x, enc_output, enc_output)
        x = self.fc_layer(x)
        return x
