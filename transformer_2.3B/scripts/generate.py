import glob
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from tokenizers import Tokenizer
import torch.nn.functional as F
from model_f.model import Model
from model_f.config import Config

root = os.path.dirname(os.path.dirname(__file__))


def find_latest_checkpoint():
    """Return the path to the highest-numbered model_epoch_*.pth checkpoint.

    Falls back to model.pth at the project root if no epoch checkpoints exist.
    """
    pattern = os.path.join(root, 'model_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    if checkpoints:
        def epoch_num(path):
            match = re.search(r'model_epoch_(\d+)\.pth$', os.path.basename(path))
            return int(match.group(1)) if match else -1
        return max(checkpoints, key=epoch_num)

    fallback = os.path.join(root, 'model.pth')
    if os.path.exists(fallback):
        return fallback

    raise FileNotFoundError(
        'No checkpoint found. Expected model_epoch_*.pth or model.pth in '
        f'{root}. Run scripts/train_v2.py first.'
    )


def build_training_config(state_dict):
    """Recreate the small config used by the training scripts.

    The vocab size is inferred directly from the checkpoint so the model
    always matches the saved weights regardless of which tokenizer was used.
    """
    config = Config()
    config.n_layers = 1
    config.dim = 128
    config.n_heads = 4
    config.kv_heads = 1
    config.dim_ff = 512
    config.max_seq_len = 256

    # Infer vocab size from the token embedding weight shape so loading the
    # checkpoint never fails on a size mismatch.
    emb_weight = state_dict.get('token_emb.weight')
    if emb_weight is not None:
        config.vocab_size = emb_weight.shape[0]
    return config


def load_model():
    checkpoint_path = find_latest_checkpoint()
    print(f'Loading checkpoint: {checkpoint_path}')

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    config = build_training_config(state_dict)

    model = Model(config)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()
tokenizer = Tokenizer.from_file(os.path.join(root, 'tokenizer', 'jarvis_tokenizer.json'))

def generate(prompt, max_len=100, temperature=0.8):
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor([tokens], device=device)

    for _ in range(max_len):
        # Keep the context within the model's max sequence length.
        x_cond = x[:, -model.config.max_seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    return tokenizer.decode(x[0].tolist())

if __name__ == '__main__':
    print(generate('Hello, how are you?'))