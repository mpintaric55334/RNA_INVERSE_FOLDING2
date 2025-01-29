from torchmetrics import Metric
import torch


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, preds, true, mask):
        mask = mask[:, 1:]
        preds = preds[:, 1:]
        true = true[:, 1:]
        true = true * mask
        self.correct += torch.sum((preds == true) & (true != 0))
        self.total += torch.sum(true != 0)

    def compute(self):
        return self.correct.float()/self.total.float()

    def reset(self):
        self.correct = torch.tensor(0, dtype=torch.float, device="cuda")
        self.total = torch.tensor(0, dtype=torch.float, device="cuda")
