import torch
import numpy as np
import torch.nn as nn


def augment_train_set(train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset):

    np.random.seed(0)
    random_indices = np.random.choice(len(valid_dataset), 5000, replace=False)
    random_valid_set = torch.utils.data.Subset(valid_dataset, random_indices)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, random_valid_set])

    return train_dataset

def mixup_data(x, y, alpha=0.2):
    """
    Applies Mixup Augmentation to the given data
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup Loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CESmoothingLoss(nn.Module):
    def __init__(self,smoothing=0.1):
        super(CESmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        with torch.no_grad():
            weight = pred.new_ones(pred.size()) * (self.smoothing / (pred.size(-1) - 1.))
            weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

