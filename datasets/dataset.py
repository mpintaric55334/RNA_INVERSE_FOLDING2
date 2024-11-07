import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tokenizer import Tokenizer
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


def parse_bpseq(filename: str, cutoff_size: int):

    """
    Function that parses the bpseq file, creating
    the adjacency matrix and generating the sequence
    for every file

    Arguments:
        - filename: str => name of the bpseq file
        - cutoff_size: int => max size of sequence for computational purposes

    Returns:
        - adjacency_matrix: torch.Tensor => binary adjacency matrix of
          secondary structure
        - sequence: str => RNA sequence represented as a string
        - valid: bool => returns if sequence is valid for usage or not
    """
    pairs = []
    sequence = ""
    valid = True
    N_count = 0  # variable which counts number of unknown nucleotides
    with open(filename, 'r') as file:

        for line in file:
            if "#" in line:
                continue  # ignore lines with #, these are description lines
            line_elements = line.strip().split()
            # properly parsed line: index, nucleotide, index of pair
            if len(line_elements) == 3:
                index, nucleotide, pair_index = line_elements
                # indices in bpseq file start from 0, so -1 is needed
                index, pair_index = int(index) - 1, int(pair_index) - 1

                nucleotide = nucleotide.upper()  # prevent upper/lowercase err

                # only canonical nucleotides for now
                if nucleotide not in "ACGU":
                    if nucleotide == "T":
                        nucleotide = "U"
                    else:
                        nucleotide = "N"
                        N_count += 1
                sequence += nucleotide

                if pair_index != -1:
                    pairs.append([index, pair_index])

        N = len(sequence)
        if N > cutoff_size:
            valid = False

        if N_count > N * 0.3:
            valid = False

        adjacency_matrix = torch.zeros((N, N), dtype=torch.float32)
        for index, pair_index in pairs:
            adjacency_matrix[index][pair_index] = 1

    return adjacency_matrix, sequence, valid


def parse_bracket_notation(secondary_structure: str):

    """
    Function that parses dot bracket secondary structure notation, exmp: (()..)
    into an adjacency matrix.

    Arguments:
        - secondary structure: str => dot bracket secondary structure notation

    Returns:
        - adjacency_matrix: torch.Tensor => binary adjacency matrix of
          secondary structure
    """
    N = len(secondary_structure)
    left_brackets = []
    adjacency_matrix = torch.zeros((N, N), dtype=torch.float32)

    for index in range(N):
        if secondary_structure[index] == "(":
            left_brackets.append(index)
        if secondary_structure[index] == ")":
            index_left = left_brackets.pop()
            adjacency_matrix[index_left][index] = 1
            # matrix needs to be mirrored
            adjacency_matrix[index][index_left] = 1
    return adjacency_matrix


def tokenize_and_mask(sequence: str):
    """
    Function that tokenizes the sequence and makes
    the aproppriate mask to be used in the loss function

    Arguments:
        - sequence: str => RNA sequence

    Returns:
        - tokenized_sequence: np.array[int] => tokenized seq with start token
        - mask: np.array[int] => mask of sequence
    """

    tokenizer = Tokenizer()
    sequence = "S" + sequence  # add start token
    tokenized_sequence = tokenizer.embedd(sequence)
    mask = []
    for element in tokenized_sequence:
        if element != 5:  # token for N and P
            mask.append(1)
        else:
            mask.append(0)

    mask = mask[1:]  # TBD if first element will be removed
    mask = np.array(mask)
    return tokenized_sequence, mask


class BPSeqDataset(Dataset):

    """
    Custom torch dataset containing position matrices and appropriate sequences

    """

    def __init__(self, bprna_directory_path: str, RnaStrAling_path: str,
                 cutoff_size: int = 256):
        self.matrices = []
        self.sequences = []
        self.masks = []
        self.true_lengths = []
        # parse bprnma
        for dirpath, _, filenames in os.walk(bprna_directory_path):
            for filename in filenames:
                if filename.endswith('.bpseq'):
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        adjacency_matrix, sequence, valid = parse_bpseq(
                            file_path, cutoff_size)
                        if not valid:
                            continue
                        N = len(sequence)
                        tokenized_sequence, mask = tokenize_and_mask(sequence)
                        self.matrices.append(adjacency_matrix)
                        self.sequences.append(tokenized_sequence)
                        self.masks.append(mask)
                        self.true_lengths.append(N)
        # parse RNAStrAlign

        for dirpath, _, filenames in os.walk(RnaStrAling_path):
            for filename in filenames:
                if filename.endswith('.bpseq'):
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        adjacency_matrix, sequence, valid = parse_bpseq(
                            file_path, cutoff_size)
                        if not valid:
                            continue
                        N = len(sequence)
                        tokenized_sequence, mask = tokenize_and_mask(sequence)
                        self.matrices.append(adjacency_matrix)
                        self.sequences.append(tokenized_sequence)
                        self.masks.append(mask)
                        self.true_lengths.append(N)
        # parse and load archiveii
        # PROVJERI JOS JEL TRIBA CISTIT ARCHIVEII

        df = pd.read_parquet("hf://datasets/multimolecule/archiveii/test.parquet")
        for _, row in df.iterrows():

            sequence = row["sequence"]
            sequence = sequence.replace("T", "U")
            sec_structure = row["secondary_structure"]
            adjacency_matrix = parse_bracket_notation(sec_structure)
            N = len(sequence)
            if N > cutoff_size:
                continue
            tokenized_sequence, mask = tokenize_and_mask(sequence)
            self.matrices.append(adjacency_matrix)
            self.sequences.append(tokenized_sequence)
            self.masks.append(mask)
            self.true_lengths.append(N)

    def __len__(self):

        return len(self.matrices)

    def __getitem__(self, idx):

        return self.matrices[idx], self.sequences[idx], self.masks[idx], self.true_lengths[idx]


class BPSeqDataModule(pl.LightningDataModule):
    def __init__(self, cutoff_size: int = 512, batch_size: int = 32,
                 split_ratio: float = 0.8):
        super().__init__()
        self.cutoff_size = cutoff_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def setup(self):
        # Create the dataset
        full_dataset = BPSeqDataset("/home/mpintaric/RNA_FOLDING/all_data/bpRNA_1m_90_dataset/bpRNA_1m_90_BPSEQLFILES",
                                    "/home/mpintaric/RNA_FOLDING/all_data/RNAStrAlign/RNAStrAlign_bpseq",
                                    cutoff_size=self.cutoff_size)
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset,
                                                            [train_size,
                                                             val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # placeholder
        return DataLoader(self.val_dataset, batch_size=self.batch_size)