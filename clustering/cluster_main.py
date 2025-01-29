from cluster_func import generate_fasta_for_cluster
MAX_SEQUENCE_LENGTH = 35


generate_fasta_for_cluster("data/output.fasta",
                           cutoff_size=MAX_SEQUENCE_LENGTH)