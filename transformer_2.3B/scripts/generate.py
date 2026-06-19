import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from tokenizers import Tokenizer
import torch.nn.functional as F
from model_f.model import Model
from model_f.config import Config

root = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(root, 'model.pth')

def load_model():
    config = Config()
    model = Model(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
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
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    return tokenizer.decode(x[0].tolist())

if __name__ == '__main__':
    print(generate('Hello, how are you?'))