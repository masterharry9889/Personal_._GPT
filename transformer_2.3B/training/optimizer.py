import torch
from torch.optim import AdamW

def get_optimizer(model, lr=3e-4, weight_decay=0.01):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)