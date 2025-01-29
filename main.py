from model.model import RNAModel
from datasets.dataset import RNADataModule
import pytorch_lightning as pl
from model.py_lightning_model import ThunderModel
import torch
import random
from clustering.cluster_func import generate_train_test_id
import wandb
from pytorch_lightning.loggers import WandbLogger


wandb_logger = WandbLogger(project="RNA SS Inverse")
random.seed(0)
torch.manual_seed(0)

MAX_SEQUENCE_LENGTH = 200

train_id, val_id, test_id = generate_train_test_id("/home/mpintaric/RNA_FOLDING/RNA_INVERSE_FOLDING2/clustering/data/clust_cluster.tsv",
                                                   0.6, 0.2)

model = RNAModel(60, 3, 40, "cuda", 7, 6, 10, 80, 3, 1, 0.0, False, 6, 4, 0.0, MAX_SEQUENCE_LENGTH).to("cuda")

dm = RNADataModule(train_id, val_id, test_id, 2, 0.8, MAX_SEQUENCE_LENGTH)

thunder = ThunderModel(model, MAX_SEQUENCE_LENGTH)
trainer = pl.Trainer(accelerator="gpu", devices=[0], min_epochs=10,
                     max_epochs=1000, check_val_every_n_epoch=5,
                     logger=wandb_logger)
trainer.fit(thunder, dm)
trainer.predict(thunder, dm)
acc = thunder.accuracy.compute()
wandb_logger.experiment.log({"test_accuracy": acc})
print(acc)
