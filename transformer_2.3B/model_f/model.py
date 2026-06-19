import torch
import torch.nn as nn
from model_f.config import Config
from model_f.block import TransformerBlock
from model_f.rope import build_rope_cache

class Model(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config if config is not None else Config()

        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.norm = nn.Linear(self.config.dim, self.config.dim)
        self.head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)

        self.rope_sin = None
        self.rope_cos = None

    def build_rope(self, device):
        sin, cos = build_rope_cache(self.config.max_seq_len, self.config.dim // self.config.n_heads, device)
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