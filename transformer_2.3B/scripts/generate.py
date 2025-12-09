import torch
from tokenizer import Tokenizer
from model_f.model import Model
from model_f.config import Config

# Load model and Tokenizer
model = Model(Config)
model.load_state_dict(torch.load("model.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer.from_file("jarvis_tokenizer.json")

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

print(generate("Hello, how are you?"))