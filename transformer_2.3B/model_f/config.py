class Config:
    vocab_size = 200_000
    max_seq_len = 4096
    dim = 4096
    n_layers = 48
    n_heads = 32
    kv_heads = 4
    rope_theta = 10000
    dim_ff = 14336
    dropout = 0.0
    bias = False
    parallel_residual = True