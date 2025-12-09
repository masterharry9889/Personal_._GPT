from re import X
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forword(self, x):
        norm = X.norm(2, dim=-1, keepdim=True)
        return X / (norm + self.eps) * self.weight
