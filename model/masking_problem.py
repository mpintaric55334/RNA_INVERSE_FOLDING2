import torch
import torch.nn as nn
import math


def dot_product_attention(q, k, v, attn_mask=None, dropout=None,
                          attention_1d=False):

    c = q.shape[-1]
    attn = torch.matmul(q, k.transpose(-1, -2))
    if attn_mask is not None:
        if attention_1d:
            attn_mask = attn_mask.unsqueeze(1)
        attn = attn.masked_fill(attn_mask == 0, -1e34)
        #  i use -1e34 to solve nan problem
    attn = attn.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAxialAttention(nn.Module):

    def __init__(self, c_in: int, num_heads: int, row: bool = True,
                 attention_dropout: float = 0.0, bias: bool = False):

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

        self.attention_dropout = nn.Dropout(p=attention_dropout)

    def forward(self, q, k, v, attn_mask=None):

        q1 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        k1 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        v1 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        
        q2 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        k2 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        v2 = torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]],
                           [[13, 14], [15, 16], [17, 18]]])
        
        q = torch.stack((q1,q2), dim=0)
        k = torch.stack((k1,k2), dim=0)
        v = torch.stack((v1,v2), dim=0)
        
        bs = q.shape[0]  # batch size
        N = q.shape[1]

        q = q.view(bs, N, N, self.num_heads,
                              self.c_head).transpose(-2, -3)
        k = q.view(bs, N, N, self.num_heads,
                              self.c_head).transpose(-2, -3)
        v = q.view(bs, N, N, self.num_heads,
                              self.c_head).transpose(-2, -3)

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, None, None, :]
        if not self.row:
            q = q.transpose(-2, -4)
            k = k.transpose(-2, -4)
            v = v.transpose(-2, -4)
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(-1, -4)
            

        output, attn = dot_product_attention(q, k, v, attn_mask,
                                             self.attention_dropout)

        if not self.row:
            output = output.transpose(-2, -4)

        output = output.transpose(-2, -3).contiguous().view(bs, N, N,
                                                            self.num_heads *
                                                            self.c_head)
        
        return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, c_in: int, num_heads: int,
                 attention_dropout: float = 0.0, bias: bool = False):

        super().__init__()
        assert c_in % num_heads == 0, (
            "Embedding dimensionality must be divisible with number"
            "of attention heads!"
        )
        self.c_in = c_in
        self.num_heads = num_heads

        self.c_head = c_in // self.num_heads
        self.c_qkv = self.c_head * num_heads

        self.attention_dropout = nn.Dropout(p=attention_dropout)

    def forward(self, attn_mask=None):

        q1 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        k1 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        v1 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        q2 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        k2 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        v2 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        q3 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        k3 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        v3 = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        q = torch.stack((q1,q2,q3), dim=0)
        k = torch.stack((k1,k2,k3), dim=0)
        v = torch.stack((v1,v2,v3), dim=0)
        bs = q.shape[0]
        q = q.view(bs, -1, self.num_heads,
                              self.c_head).transpose(-2, -3)
        k = k.view(bs, -1,
                              self.num_heads, self.c_head).transpose(-2, -3)
        v = v.view(bs, -1,
                              self.num_heads, self.c_head).transpose(-2, -3)
        output, attn = dot_product_attention(q, k, v, attn_mask,
                                             self.attention_dropout,
                                             attention_1d=True)
        output = output.transpose(-2, -3).contiguous().view(bs, -1,
                                                            self.num_heads *
                                                            self.c_head)
        return output, attn



mha = MultiHeadAttention(2,2)
mask = torch.randint(0,2,(3,4,4))
print(mask)
out,_ = mha(mask)
print(out.shape)
