import pytorch_lightning as pl
from model.model import RNAModel
from model.loss import Loss
from torch.optim import Adam
import torch
from model.metrics import Accuracy


class ThunderModel(pl.LightningModule):

    def __init__(self, rna_model: RNAModel, max_length: int):
        super().__init__()
        self.rna_model = rna_model
        self.loss_crit = Loss()
        self.accuracy = Accuracy()

    def forward(self, matrix, seq, attn_mask_enc=None, attn_mask_dec_mha=None,
                attn_mask_dec_csat=None):

        out = self.rna_model(matrix, seq, attn_mask_enc, attn_mask_dec_mha,
                             attn_mask_dec_csat)
        return out

    def _common_loss_step(self, batch, batch_idx):
        seq, matr, loss_mask, dec_csat_mask, dec_mha_masks, enc_masks = batch
        pred = self(matr, seq, enc_masks, dec_mha_masks, dec_csat_mask)
        loss = self.loss_crit.compute_loss(pred, seq, loss_mask)
        return loss

    def _common_test_step(self, batch, batch_idx):
        tr_seq, matr, loss_mask, _, _, enc_masks = batch
        # encoder part
        matr = self.rna_model.embedding_matrix(matr)
        enc_output = self.rna_model.encoder(matr, enc_masks)
        batch_size = tr_seq.shape[0]
        N = tr_seq.shape[1]
        start_tokens = torch.full((batch_size, 1), 6, dtype=torch.long, device="cuda")
        for _ in range(1, N):
            seq = self.rna_model.embedding_sequence(start_tokens)
            seq = self.rna_model.pos_encoding(seq)
            preds = self.rna_model.decoder(seq, enc_output, None,
                                           None)
            preds = torch.argmax(preds, dim=2)[:, -1].unsqueeze(1) + 1
            start_tokens = torch.cat((start_tokens, preds), dim=1)
        self.accuracy.update(start_tokens, tr_seq, loss_mask)

    def training_step(self, batch, batch_idx):
        loss = self._common_loss_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, on_step=False, on_epoch=True,
                      prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_loss_step(batch, batch_idx)
        self.log_dict({"validation_loss": loss}, on_step=False, on_epoch=True,
                      prog_bar=True)
        self._common_test_step(batch, batch_idx)
        return loss

    def on_validation_epoch_end(self):

        accuracy = self.accuracy.compute()

        self.log("validation_accuracy", accuracy, on_step=False, on_epoch=True,
                 prog_bar=True)

        self.accuracy.reset()

    def predict_step(self, batch, batch_idx):
        self._common_test_step(batch, batch_idx)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)
