import torch
from torch.utils.data import Dataset
import os
from tokenizer import Tokenizer
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


"""
I need to figure out a way to make this setting global, or to make it callable in pad_collate_fn
"""
MAX_SEQUENCE_LENGTH = 128


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

        adjacency_matrix = torch.ones((N, N), dtype=torch.float32)
        for index, pair_index in pairs:
            adjacency_matrix[index][pair_index] = 2

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
    adjacency_matrix = torch.ones((N, N), dtype=torch.float32)

    for index in range(N):
        if secondary_structure[index] == "(":
            left_brackets.append(index)
        if secondary_structure[index] == ")":
            index_left = left_brackets.pop()
            adjacency_matrix[index_left][index] = 2
            # matrix needs to be mirrored
            adjacency_matrix[index][index_left] = 2
    return adjacency_matrix


def tokenize(sequence: str):
    """
    Function that tokenizes the sequence.

    Arguments:
        - sequence: str => RNA sequence

    Returns:
        - tokenized_sequence: np.array[int] => tokenized seq with start token
    """

    tokenizer = Tokenizer()
    sequence = "S" + sequence  # add start token
    tokenized_sequence = tokenizer.embedd(sequence)
    return tokenized_sequence


class bpRNADataset(Dataset):
    """
    Custom torch dataset for bpRNADataset
    """
    def __init__(self, bprna_directory_path: str, cutoff_size: int = 256):
        """
        Arguments:
            - bprna_directory_path: str => path to bprna directory
            - cutoff_size: int => max size of sequence
        """
        self.matrices = []
        self.sequences = []
        for dirpath, _, filenames in os.walk(bprna_directory_path):
            for filename in filenames:
                if filename.endswith('.bpseq'):
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        adjacency_matrix, sequence, valid = parse_bpseq(
                            file_path, cutoff_size)
                        if not valid:
                            continue
                        tokenized_sequence = tokenize(sequence)
                        self.matrices.append(adjacency_matrix)
                        self.sequences.append(tokenized_sequence)

    def __len__(self):

        return len(self.matrices)

    def __getitem__(self, idx):

        return self.matrices[idx], self.sequences[idx]


class RNAStrAlignDataset(Dataset): 
    """
    Custom torch dataset for RNAStrAlign.

    """

    def __init__(self, RnaStrAlign_path: str, cutoff_size: int = 256):

        """
        Arguments:
            - RnaStrAlign_path: str => path to bRnaStrAlign directory
            - cutoff_size: int => max size of sequence
        """
        self.matrices = []
        self.sequences = []

        for dirpath, _, filenames in os.walk(RnaStrAlign_path):
            for filename in filenames:
                if filename.endswith('.bpseq'):
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path):
                        adjacency_matrix, sequence, valid = parse_bpseq(
                            file_path, cutoff_size)
                        if not valid:
                            continue
                        tokenized_sequence = tokenize(sequence)
                        self.matrices.append(adjacency_matrix)
                        self.sequences.append(tokenized_sequence)

    def __len__(self):

        return len(self.matrices)

    def __getitem__(self, idx):

        return self.matrices[idx], self.sequences[idx]


class ArchiveiiDataset(Dataset):
    """
    Custom torch dataset for ArchiveiiDataset.
    """

    def __init__(self, cutoff_size: int = 256):
        """
        Arguments:
            - cutoff_size: int => max size of sequence
        """
        self.matrices = []
        self.sequences = []
        df = pd.read_parquet("hf://datasets/multimolecule/archiveii/test.parquet")
        for _, row in df.iterrows():

            sequence = row["sequence"]
            sequence = sequence.replace("T", "U")
            sec_structure = row["secondary_structure"]
            adjacency_matrix = parse_bracket_notation(sec_structure)
            N = len(sequence)
            if N > cutoff_size:
                continue
            tokenized_sequence, mask = tokenize(sequence)
            self.matrices.append(adjacency_matrix)
            self.sequences.append(tokenized_sequence)

    def __len__(self):

        return len(self.matrices)

    def __getitem__(self, idx):

        return self.matrices[idx], self.sequences[idx]


def pad_collate_fn(batch):
    """
    Custom collate_fn. Creates paddings for sequences and adjacency matrices,
    and also creates masks for encoder, decoder, and loss calculation.
    """
    padded_sequences = []
    loss_masks = []
    encoder_masks = []
    decoder_masks = []
    padded_matrices = []
    for matrix, seq in batch:
        # pad sequence
        pad_seq_temp = torch.Tensor(seq)
        pad_seq = torch.zeros(MAX_SEQUENCE_LENGTH)
        N = len(pad_seq_temp)
        pad_seq[:N] = pad_seq_temp
        padded_sequences.append(pad_seq)
        # create loss_mask
        loss_mask = []
        for nucl in pad_seq:
            if nucl not in (0, 5):  # nucleotide not padding or unknown
                loss_mask.append(1)
            else:
                loss_mask.append(0)
        loss_masks.append(torch.Tensor(loss_mask))
        # create encoder mask
        encoder_mask = torch.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
        encoder_mask[:N, :N] = torch.ones((N, N))
        encoder_masks.append(encoder_mask)
        # create decoder mask
        decoder_mask = torch.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
        decoder_mask[:N, :N] = torch.tril(torch.ones(N, N, dtype=torch.bool),
                                          diagonal=0)
        decoder_masks.append(decoder_mask)
        # pad matrix
        padded_matrix = torch.zeros((MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH))
        padded_matrix[:N, :N] = torch.Tensor(matrix)
        padded_matrices.append(padded_matrix)

    padded_sequences = torch.stack(padded_sequences)
    loss_masks = torch.stack(loss_masks)
    encoder_masks = torch.stack(encoder_masks)
    decoder_masks = torch.stack(decoder_masks)
    padded_matrices = torch.stack(padded_matrices)

    return (
        padded_sequences, padded_matrices, loss_masks,
        encoder_masks, decoder_masks
    )


class BPSeqDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 32, split_ratio: float = 0.8,
                 cutoff_size: int = MAX_SEQUENCE_LENGTH,):
        super().__init__()
        self.cutoff_size = cutoff_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def setup(self):
        # Create the dataset
        full_dataset = ArchiveiiDataset(cutoff_size=self.cutoff_size)
        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset,
                                                            [train_size,
                                                             val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, collate_fn=pad_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=pad_collate_fn)

    def test_dataloader(self):
        # placeholder
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=pad_collate_fn)