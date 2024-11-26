import torch
import torch.nn as nn
from attention import MultiHeadAxialAttention


class ConvLayer(nn.Module):
    """
    This class implements the convolution layer
    of the encoder block.

    Link to RNAinformer conv layer if you want to see what
    they did:
    https://anonymous.4open.science/r/RNA-design-7204/RNAinformer/modules/feed_forward.py
    """

    def __init__(self, embed_size: int, kernel_size: int = 3,
                 out_channels: int = 256, padding="same"):
        """
        Arguments:
            - embed_size: embedding size of the model,
                          represents the input channel number
            - kernel_size: kernel size of the convolution
            - out_channels: number of out channels of the convolution
            - padding: type of padding, preferably always same
        """
        super(ConvLayer, self).__init__()

        """
        RNAInformer uses GroupNorm, will have some questions
        """
        self.norm1 = nn.GroupNorm(1, embed_size)
        self.conv1 = nn.Conv2d(in_channels=embed_size,
                               out_channels=out_channels,
                               kernel_size=kernel_size, padding=padding)
        # SiLU is used in RNAInformer, so why not try?
        self.act1 = nn.SiLU()

        # they also didnt use second normalisation, so i didnt too
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=embed_size,
                               kernel_size=kernel_size, padding=padding)
        """
        RNAInformer doesnt use activation function after second layer,
        I didnt like it, so I used it
        """
        self.act2 = nn.SiLU()

    def forward(self, x):
        """
        I added residual connection to mimic what we did in last semester,
        they also do residuals
        """
        """
        For convolution to work, need to reshape data from
        (bs, N, N, embedd) to
        (bs, embedd, N, N), also, needed to use 2d conv,
        where embedding dim is actually channels??,
        whats the deal with that, is there some
        paper or article which explains this in more
        detail, or is this one of it works, we dont know why things
        in ML
        """
        _x = x
        x = x.permute(0, 3, 1, 2)
        # does normalisation go before or after i remeber residual connections?
        x = self.norm1(x)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = x.permute(0, 2, 3, 1)
        x = x + _x
        return x


class AxialAttentionLayer(nn.Module):
    """
    Implementation of axial attention layer,
    which goes inside the encoder block.
    """

    def __init__(self, embedd_size: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = False):
        """
        Arguments:
            - embedd_size: embedding dimension of the model
            - num_heads: number of heads for attention
            - dropout: attention dropout
            - bias: should bias be added to attention
        """

        """
        In RNAInformer they divide dropout by 2 for
        row and column attention, is that the mainstream way
        of distributing dropout by rows and columns in axial??
        Also, if I have dropout in my attention class is it even necessary here
        (i have same dropout inside attention class and after class output)
        """
        super(AxialAttentionLayer, self).__init__()
        self.dropout1 = nn.Dropout(dropout / 2)
        self.dropout2 = nn.Dropout(dropout / 2)
        self.norm1 = nn.LayerNorm(embedd_size)
        self.norm2 = nn.LayerNorm(embedd_size)
        self.attention_row = MultiHeadAxialAttention(c_in=embedd_size,
                                                     num_heads=num_heads,
                                                     row=True,
                                                     attention_dropout=dropout/2,
                                                     bias=bias)
        self.attention_column = MultiHeadAxialAttention(c_in=embedd_size,
                                                        num_heads=num_heads,
                                                        row=False,
                                                        attention_dropout=dropout/2,
                                                        bias=bias)

    def forward(self, x, attn_mask=None):
        _x = x
        x = self.norm1(x)
        x, _ = self.attention_row(x, x, x, attn_mask)
        x = self.dropout1(x)
        x = x + _x
        _x = self.norm2(x)
        x, _ = self.attention_column(x, x, x, attn_mask)
        x = self.dropout2(x)
        x = x + _x
        return x


class EncoderBlock(nn.Module):
    """
    Implementation of one encoder block of
    the entire encoder.
    One block consists of axial row, axial column,
    and convolution layer.
    """

    def __init__(self, embedd_size: int, num_heads: int,
                 conv_out_channels: int,
                 kernel_size_conv: int,
                 bias: bool = False,
                 attention_dropout: float = 0.0,
                 padding: str = "same"):
        """
        Arguments:
            - embedd_size: embedding dim of the model
            - num_heads: number of attention heads
            - conv_out_channels: number of out channels for the middle
            of the convolution layer
            - kernel_size_conv: size of convolution_kernel
            - bias: should bias be added to attention 
            - attention_dropout: attention dropout
            - padding: padding for conv, for now
             only same works
        """
        super(EncoderBlock, self).__init__()
        self.attention_layer = AxialAttentionLayer(embedd_size, num_heads,
                                                   attention_dropout,
                                                   bias=bias)
        self.convolution_layer = ConvLayer(embed_size=embedd_size,
                                           kernel_size=kernel_size_conv,
                                           out_channels=conv_out_channels,
                                           padding=padding)
        # should I add another dropout?

    def forward(self, x, attn_mask=None):
        x = self.attention_layer(x, attn_mask)
        x = self.convolution_layer(x)
        return x


class EncoderOutputReduction:

    """
    Class that implements the final reduction of encoder.
    After passing through the encoder block, its output
    is of shape (batch_size, N, N, embedd_size).
    This is not compatible
    """

    def __init__(self, reduction: int | tuple[int, int]):
        """
        Arguments:
            - reduction: type of encoder output reduction,
            1: row vise
            2: columnn wise
            (1,2): row and column wise (global average)
        """
        if reduction not in [1, 2, (1, 2)]:
            raise ValueError("Invalid reduction technique")
        self.reduction = reduction

    def reduce(self, x, mask=None):

        # small constant to evade division by zero
        eps = 1e-34

        if mask is not None:
            # before unsqueeze mask shape (bs,N,N)
            mask = mask.unsqueeze(-1)
            # after shape is (bs,N,N,1), compatible for pointwise multip
            x = x * mask
            x = torch.sum(x, self.reduction) / (
                eps + torch.sum(mask, self.reduction))
        else:
            x = torch.mean(x, self.reduction)

        return x


class Encoder(nn.Module):
    """
    Class that implements the full encoder model
    """

    def __init__(self, num_blocks: int, embedd_size: int, num_heads: int,
                 conv_out_channels: int, kernel_size_conv: int,
                 reduction: tuple, attention_dropout: float = 0.0,
                 bias=False, padding="same"):
        """
        Arguments:
            - num_blocks: number of encoder blocks
            - embedd_size: embedding dim of the model
            - num_heads: number of attention heads
            - conv_out_channels: number of out channels for the middle
            of the convolution layer
            - kernel_size_conv: size of convolution_kernel
            - reduction: type of encoder output reduction
            - attention_dropout: attention dropout
            - bias: should bias be added to attention
            - padding: padding for conv, for now
             only same works
        """
        super(Encoder, self).__init__()

        self.blocks = nn.ModuleList([
            EncoderBlock(embedd_size, num_heads, conv_out_channels,
                         kernel_size_conv, bias,
                         attention_dropout,
                         padding=padding) for _ in range(num_blocks)
        ])
        self.reduction = EncoderOutputReduction(reduction)

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.reduction.reduce(x, attn_mask)
        return x
