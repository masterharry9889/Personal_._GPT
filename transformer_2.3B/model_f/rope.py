import torch 

def apply_rope(x, sin, cos):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return (x1 * cos) + (x2 * sin), (x2 * cos) - (x1 * sin)

def build_rope_cache(seq_len, head_dim, device, theta = 10000):
    pos = torch.arange(seq_len, device=device).float()
    dim = torch.arange(head_dim, device=device).float()

    freqs = 1.0 / (theta ** (dim / head_dim))
    angles = pos[:, None] * freqs[None, :]
    return torch.sin(angles), torch.cos(angles)