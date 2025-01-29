import pandas as pd
import random


def generate_fasta_for_cluster(output_file: str, cutoff_size: int = 50):
    """
    Function that takes a data source and converts it into a fasta file.
    This is needed for MMseqs2 clustering.

    Arguments:
        - output_file: str => name of the fasta file
        - cutoff_size: int => max size of sequence for computational purposes

    """
    df = pd.read_parquet("/home/mpintaric/RNA_FOLDING/all_data/Archiveii/test.parquet")
    sequences = []

    for _, row in df.iterrows():
        id = row["id"]
        sequence = row["sequence"]
        sequence = sequence.replace("T", "U")
        N = len(sequence)
        if N > cutoff_size - 1:  # important to prevent overflow, discuss later
            continue
        sequences.append((id, sequence))

    with open(output_file, "w") as fasta_file:
        for seq_id, sequence in sequences:
            fasta_file.write(f">{seq_id}\n{sequence}\n")


def generate_train_test_id(cluster_tsv: str, train_ratio: float = 0.7,
                           val_ratio: float = 0.15):
    """
    Function that takes the cluster tsv file and generates
    id's for train, val and test.

    Arguments:
        - cluster_tsv: str => name of the tsv cluster file
        - train_ratio: float => proportion of train data
        - val_ratio: float => proportion of val data

    Returns:
        - train_id, val_id, test_id: list[str] => list of chosen id's
    """
    clusters = {}
    with open(cluster_tsv, "r") as file:
        for line in file.readlines():
            line = line.strip().split("\t")
            if line[0] not in clusters:
                clusters[line[0]] = [line[1]]
            else:
                temp = clusters[line[0]]
                temp.append(line[1])
                clusters[line[0]] = temp

    n_train = int(train_ratio * len(clusters))
    n_valid = int(val_ratio * len(clusters))
    n_test = len(clusters) - n_train - n_valid
    clusters_cp = clusters.copy()

    train_keys = random.sample(list(clusters.keys()), n_train)
    for key in train_keys:
        del clusters[key]

    val_keys = random.sample(list(clusters.keys()), n_valid)
    for key in val_keys:
        del clusters[key]

    test_keys = random.sample(list(clusters.keys()), n_test)
    for key in test_keys:
        del clusters[key]

    train_id = []
    val_id = []
    test_id = []

    for key in train_keys:
        train_id += clusters_cp[key]
    for key in val_keys:
        val_id += clusters_cp[key]
    for key in test_keys:
        test_id += clusters_cp[key]

    return train_id, val_id, test_id
