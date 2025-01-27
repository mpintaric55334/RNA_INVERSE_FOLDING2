from model.model import RNAModel
from datasets.dataset import RNADataModule
import pytorch_lightning as pl
from model.py_lightning_model import ThunderModel
import torch
from model.metrics import Accuracy

MAX_SEQUENCE_LENGTH = 200



model = RNAModel(60, 3, 40, "cuda", 7, 6, 10, 80, 3, 1, 0.0, False, 6, 4, 0.0, MAX_SEQUENCE_LENGTH).to("cuda")

dm = RNADataModule(2, 0.8, MAX_SEQUENCE_LENGTH)
thunder = ThunderModel(model, MAX_SEQUENCE_LENGTH)
trainer = pl.Trainer(accelerator="gpu", devices=[0], min_epochs=10,
                     max_epochs=100, check_val_every_n_epoch=5)
trainer.fit(thunder, dm)
trainer.predict(thunder, dm)
acc = thunder.accuracy.compute()
print(acc)
