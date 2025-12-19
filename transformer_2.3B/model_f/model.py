import torch
import torch.nn as nn
from model_f.config import Config
from model_f.block import TransformerBlock
from model_f.rope import build_rope_cache

class Model(nn.Module):
    def __init__(self, config = Config()):
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(Config) for _ in range(Config.n_layers)])
        self.norm = nn.Linear(Config.dim, Config.dim)
        self.head = nn.Linear(Config.dim, Config.vocab_size, bias=False)

        self.register_buffer("rope_sin", None)
        self.register_buffer("rope_cos", None)

    def build_rope(self, device):
        sin, cos = build_rope_cache(self.Config.max_seq_len, self.Config.dim // self.Config.n_heads, device)
        self.rope_sin = sin
        self.rope_cos = cos

    def forward(self, idx):
        B, T = idx.shape
        if self.rope_sin is None:
            self.build_rope(idx.device)

        x = self.token_emb(idx)

        for block in self.blocks:
            x = block(x, self.rope_sin[:T], self.rope_cos[:T])

        x = self.norm(x)
        return self.head(x)