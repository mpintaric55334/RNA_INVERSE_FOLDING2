import torch
import torch.nn as nn
import numpy as np


class Embedder(nn.Module):
    """
    Class that embedds adjacency matrix using Relative Positional
    Encoding and built-in torch nn.Embedding

    Expected input size of adjacency matrix: (batch, N, N)
    Expected output size: (batch, N, N, embedding_dim)
    """
    def __init__(self, N: int, embedd_size: int, edge_types: int,
                 bin_size: int):
        """
        Arguments:
            - N: int => dimensions of ajdacency matrix
            - embedd_size: int => embedding dimension
            - edge_types: int => number of distinct values
            (distinct edge types) in the adjacency matrix.
            2 if binary adjacency matrix, more if otherwise.
            - bin_size: int => the binning in relative positional
            encoding goes from [-bin_size, bin_size]
        """
        super(Embedder, self).__init__()

        # initial embedder of the adjacency matrix
        self.embedding_binary = nn.Embedding(num_embeddings=edge_types,
                                             embedding_dim=embedd_size)
        # embedder of the relative positional matrix
        self.embedding_positional = nn.Embedding(num_embeddings=2*bin_size+1,
                                                 embedding_dim=embedd_size)

        # creation of relative positional matrix
        indices = torch.arange(N)
        relative_positions = indices[None, :] - indices[:, None]
        self.positional_encode = torch.clamp(relative_positions, -bin_size,
                                             bin_size)
        self.positional_encode = self.positional_encode + bin_size

    def forward(self, x):
        binary_embedd = self.embedding_binary(x)
        positional_embedd = self.embedding_positional(self.positional_encode)
        out = binary_embedd + positional_embedd
        return out
