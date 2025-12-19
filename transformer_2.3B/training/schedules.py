def cosine_schedule(optimizer, steps, total_steps, lr_min=1e-5, lr_max=3e-4):
    import math
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * steps / total_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr