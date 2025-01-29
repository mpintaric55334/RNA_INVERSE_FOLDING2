import numpy as np
import torch


class Tokenizer:

    def __init__(self):
        # start_token = 0
        self.lookup = {"S": 6, "A": 1, "C": 2, "G": 3, "U": 4, "N": 5}

    def embedd(self, sequence):
        embedding_array = np.array([], dtype=int)

        for nucleotide in sequence:
            embedding_array = np.append(embedding_array,
                                        self.lookup[nucleotide])

        return torch.Tensor(embedding_array)
