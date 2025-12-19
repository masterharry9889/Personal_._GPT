import torch
import torch.nn as nn
from model_f.rope import apply_rope

class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // config.n_heads * config.kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, dim // config.n_head * config.kv_heads, bias=False)

        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.n_heads = config.n_heads
        self.kv_heads = config.kv_heads
        self.head_dim = dim // config.n_heads

    def forword(self, x, rope_sin, rope_cos, mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim)

        # apply rope
        q1, q2 = apply_rope(q, rope_sin, rope_cos)
        k1, k2 = apply_rope(k, rope_sin, rope_cos)

        q = torch.stack((q1, q2), dim=-1).reshape(B, T, self.n_heads, self.head_dim)
        k = torch.stack((k1, k2), dim=-1).reshape(B, T, self.kv_heads, self.head_dim)

        # Expand k,v for multi-head sharing
        k = k.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.n_heads // self.kv_heads, dim=2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)

        att = torch.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.o_proj