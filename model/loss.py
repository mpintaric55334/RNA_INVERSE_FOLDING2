import torch.nn as nn


class Loss:

    def __init__(self):

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, predictions, true_sequence, masks):
        masks = masks[:, 1:]  # shift for start token
        predictions = predictions[:, :-1, :]  # we dont care about last prediction
        predictions = predictions.permute(0, 2, 1)  # permute for compatibility with loss class
        true_sequence = true_sequence[:, 1:]  # remove first element <s>
        true_sequence = abs(true_sequence - 1)  # have to remove one, because token 1 corresponds to index 0 in probability

        loss = self.criterion(predictions, true_sequence)
        loss = loss * masks  # element wise for masking N and P
        return loss.sum() / (masks.sum() + 1e-8) # divide by sum instead of mean because of 0
