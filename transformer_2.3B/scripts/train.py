import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from model_f.model import Model
from model_f.config import Config
from training.dataset import CausalTextDataset
from training.optimizer import get_optimizer
from training.schedules import cosine_schedule
from training.trainer import Trainer


def load_text_from_csv(path, text_column='dialog'):
    texts = []
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get(text_column)
            if value:
                texts.append(value.strip())
    return '\n'.join(texts)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    tokenizer_path = os.path.join(root, 'tokenizer', 'jarvis_tokenizer.json')
    csv_path = os.path.join(root, 'data', 'train.csv')

    tokenizer = Tokenizer.from_file(tokenizer_path)
    text = load_text_from_csv(csv_path)
    try:
        tokens = tokenizer.encode(text).ids
    except Exception as e:
        print(f'Tokenizer encode error: {e}, trying without special tokens')
        tokens = tokenizer.encode(text, add_special_tokens=False).ids

    config = Config()
    config.batch_size = 2
    config.block_size = 128
    config.n_layers = 1
    config.dim = 128
    config.n_heads = 4
    config.kv_heads = 1
    config.dim_ff = 512
    config.max_seq_len = 256
    try:
        config.vocab_size = tokenizer.get_vocab_size()
    except:
        config.vocab_size = 50000

    dataset = CausalTextDataset(tokens, block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = Model(config)
    optimizer = get_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, optimizer, device=device)

    epochs = 1
    total_steps = len(dataloader) * epochs

    for epoch in range(epochs):
        avg_loss = trainer.train_epoch(dataloader, total_steps, epoch, scheduler=cosine_schedule)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        model_path = os.path.join(root, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
