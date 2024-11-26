import torch
import torch.nn as nn


class Embedder(nn.Module):
    """
    Class that embedds adjacency matrix using Relative Positional
    Encoding and built-in torch nn.Embedding

    Expected input size of adjacency matrix: (batch, N, N)
    Expected output size: (batch, N, N, embedding_dim)
    """
    def __init__(self, embedd_size: int, edge_types: int,
                 bin_size: int, device: str):
        """
        Arguments:
            - embedd_size: int => embedding dimension
            - edge_types: int => number of distinct values
            (distinct edge types) in the adjacency matrix.
            2 if binary adjacency matrix, more if otherwise.
            - bin_size: int => the binning in relative positional
            encoding goes from [-bin_size, bin_size]
            - device: str => device of the model, serves
            to create intermediate tensors on same device
        """
        super(Embedder, self).__init__()

        self.bin_size = bin_size

        # initial embedder of the adjacency matrix
        self.embedding_binary = nn.Embedding(num_embeddings=edge_types,
                                             embedding_dim=embedd_size)
        # embedder of the relative positional matrix
        self.embedding_positional = nn.Embedding(num_embeddings=2*bin_size+1,
                                                 embedding_dim=embedd_size)

        self.device = device

    def forward(self, x):

        # creation of relative positional matrix
        N = x.shape[-1]
        indices = torch.arange(N).to(self.device)
        relative_positions = indices[None, :] - indices[:, None]
        self.positional_encode = torch.clamp(relative_positions,
                                             -self.bin_size, self.bin_size)
        self.positional_encode = self.positional_encode + self.bin_size
        binary_embedd = self.embedding_binary(x)
        positional_embedd = self.embedding_positional(self.positional_encode)
        out = binary_embedd + positional_embedd
        return out
