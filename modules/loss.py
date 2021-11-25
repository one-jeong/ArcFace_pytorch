import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(ArcFaceLoss, self).__init__()
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, y_pred, y_true):
        loss = self.cross_entropy(y_pred, y_true)
        return loss