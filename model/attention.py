import torch
import torch.nn as nn
import math


def dot_product_attention(q, k, v, attn_mask=None):
    """
    Function that computes dot product attention between
    query, key, value vectors
    """
    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(c)

    if attn_mask is not None:
        attn = attn.masked_fill(attn_mask, float("-inf"))

    attn = attn.softmax(dim=-1)

    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAxialAttention(nn.Module):
    """
    Multi head axial attention implementation. Can be
    either row or column attention. If row attention,
    row argument should be True, if column attention,
    row argument should be False.
    """

    def __init__(self, c_in: int, num_heads: int, row: bool = True,
                 bias: bool = False):
        """
        Arguments:
            - c_in: embedding dimensionality
            - num_heads: number of heads
            - row: True if row axial attention, false if column
            - bias: add or dont add bias to linear layers
        """
        super().__init__()
        assert c_in % num_heads == 0, (
             "Embedding dimensionality must be divisible with number"
             "of attention heads!"
        )
        self.c_in = c_in
        self.num_heads = num_heads
        self.row = row
        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        """
        Even though there are multiple heads, they can all
        be put through the same linear layer, and afterwards
        in forward we can separate them. By doing this everything
        is faster, and the heads do not share the weights.
        """
        self.to_q = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_k = nn.Linear(self.c_in, self.c_qkv, bias=bias)
        self.to_v = nn.Linear(self.c_in, self.c_qkv, bias=bias)

        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, q, k, v):
        bs = q.shape[0]  # batch size
        N = q.shape[1]

        """
        Firstly, vectors are changed to be of dimension
        (batch size, row_length, column_length, number of heads, head_size),
        but they are transposed to be,
        (batch_size, row_length, head_size, column_length, number of heads)
        if row attention, and
        (batch_size, column_length, head_size, row_length, number of heads)
        if column attention,
        so we can do matrix multiplication
        """
        q = self.to_q(q).view(bs, N, N, self.num_heads,
                              self.c_head).transpose(-2, -3)
        k = self.to_k(k).view(bs, N, N,                        self.num_heads, self.c_head).transpose(-2, -3)
        v = self.to_v(v).view(bs, N, N,
                              self.num_heads, self.c_head).transpose(-2, -3)

        # additional transposition needed for column attention
        if not self.row:
            q = q.transpose(-2, -4)
            k = k.transpose(-2, -4)
            v = v.transpose(-2, -4)

        output, attn = dot_product_attention(q, k, v)

        # additional transposition needed for column attention
        if not self.row:
            output = output.transpose(-2, -4)

        output = output.transpose(-2, -3).contiguous().view(bs, N, N,
                                                            self.num_heads *
                                                            self.c_head)
        output = self.out_proj(output)
        return output, attn
