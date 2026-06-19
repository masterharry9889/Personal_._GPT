import torch 

def apply_rope(x, sin, cos):
    # x has shape (B, T, n_heads, head_dim); split into even/odd pairs.
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # sin/cos have shape (T, head_dim // 2). Reshape so they broadcast across
    # the batch and head dimensions: (1, T, 1, head_dim // 2).
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]

    return (x1 * cos) + (x2 * sin), (x2 * cos) - (x1 * sin)

def build_rope_cache(seq_len, head_dim, device, theta=10000):
    pos = torch.arange(seq_len, device=device).float()
    # Only compute for half the dimensions since we split x into pairs
    dim = torch.arange(0, head_dim, 2, device=device).float()

    freqs = 1.0 / (theta ** (dim / head_dim))
    angles = pos[:, None] * freqs[None, :]
    return torch.sin(angles), torch.cos(angles)
