from turtle import forward
import torch.nn as nn
from model_f.rmsmorm import RMSNorm
from model_f.attention import MultiQueryAttention
from model_f.feedforwaord import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.dim)
        self.attn = MultiQueryAttention(config)
        self.mlp = SwiGLU(config)
        self.parallel = config.parallel_residual

    def foeword(self, x, rope_sin, rope_cos, mask=None):
        if self.parallel:
            norm = self.norm(x)
            return x + self.attn(norm, rope_sin, rope_cos, mask) + self.mlp(norm)
        else :
            x = x + self.attn(self.norm(x), rope_sin, rope_cos, mask)
            x = x + self.mlp(self.norm(x))
            return x