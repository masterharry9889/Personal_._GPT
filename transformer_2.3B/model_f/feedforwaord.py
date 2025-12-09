import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.dim_ff, bias = False)
        self.w2 = nn.Linear(config.dim, config.dim_ff, bias = False)
        self.w3 = nn.Linear(config.dim_ff, config.dim, bias = False)

    def forward(self, x):
        return self.w3(torch.nn.functional(self.w1(x)) * self.w2(x))