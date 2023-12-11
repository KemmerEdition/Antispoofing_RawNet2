import torch
import torch.nn as nn


class CMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))

    def forward(self, predicts, target, **kwargs):
        loss = self.ce_loss(predicts, target)
        return {"loss": loss}
