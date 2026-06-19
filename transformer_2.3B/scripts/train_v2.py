import csv
import json
import math
import os
import sys
import time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from torch.utils.data import DataLoader, Subset

try:
    from tokenizers import Tokenizer
    HAS_TOKENIZERS = True
except:
    HAS_TOKENIZERS = False
    Tokenizer = None

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


class SimpleCharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        class TokenIds:
            def __init__(self, ids):
                self.ids = ids
        return TokenIds([self.char_to_idx.get(c, 0) for c in text])
    
    def get_vocab_size(self):
        return self.vocab_size


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    tokenizer_path = os.path.join(root, 'tokenizer', 'jarvis_tokenizer.json')
    csv_path = os.path.join(root, 'data', 'train.csv')
    val_csv_path = os.path.join(root, 'data', 'validation.csv')

    print('Loading text from CSV...')
    text = load_text_from_csv(csv_path)
    print(f'Loaded {len(text)} characters from dataset')

    val_text = ''
    if os.path.exists(val_csv_path):
        val_text = load_text_from_csv(val_csv_path)
        print(f'Loaded {len(val_text)} characters from validation set')
    
    tokenizer = None
    tokens = None
    
    if HAS_TOKENIZERS and os.path.exists(tokenizer_path):
        print('Attempting BPE tokenizer...')
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokens = tokenizer.encode(text).ids
            print(f'Success! Tokenized to {len(tokens)} tokens using BPE tokenizer')
        except Exception as e:
            print(f'BPE tokenizer failed: {e}')
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False).ids
                print(f'Tokenized to {len(tokens)} tokens without special tokens')
            except:
                tokens = None
    
    if tokens is None:
        print('Using character-level tokenizer fallback')
        tokenizer = SimpleCharTokenizer(text)
        tokens = tokenizer.encode(text).ids
        print(f'Tokenized to {len(tokens)} tokens using character tokenizer')

    # Tokenize the validation text with the same tokenizer.
    val_tokens = None
    if val_text:
        try:
            val_tokens = tokenizer.encode(val_text).ids
            print(f'Tokenized validation set to {len(val_tokens)} tokens')
        except Exception as e:
            print(f'Validation tokenization failed: {e}')
            val_tokens = None

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
        config.vocab_size = max(tokens) + 1 if tokens else 256

    print(f'Config: vocab_size={config.vocab_size}, block_size={config.block_size}, batch_size={config.batch_size}')

    dataset = CausalTextDataset(tokens, block_size=config.block_size)

    # On CPU the full dataset is enormous, so optionally cap the number of
    # training samples per epoch to keep the run to a reasonable wall-clock time.
    # Set MAX_TRAIN_SAMPLES=0 (env var) to use the entire dataset.
    max_samples = int(os.environ.get('MAX_TRAIN_SAMPLES', '2000'))
    if max_samples and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
        print(f'Capping training set to {max_samples} samples '
              f'(set MAX_TRAIN_SAMPLES=0 to use all)')

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f'Dataset: {len(dataset)} samples, {len(dataloader)} batches')

    # Build the validation dataloader (optionally capped for CPU feasibility).
    val_dataloader = None
    if val_tokens and len(val_tokens) > config.block_size:
        val_dataset = CausalTextDataset(val_tokens, block_size=config.block_size)
        max_val_samples = int(os.environ.get('MAX_VAL_SAMPLES', '500'))
        if max_val_samples and len(val_dataset) > max_val_samples:
            val_dataset = Subset(val_dataset, list(range(max_val_samples)))
            print(f'Capping validation set to {max_val_samples} samples '
                  f'(set MAX_VAL_SAMPLES=0 to use all)')
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        print(f'Validation: {len(val_dataset)} samples, {len(val_dataloader)} batches')
    else:
        print('No usable validation set found; skipping validation evaluation')

    model = Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model created with {total_params:,} parameters')
    
    optimizer = get_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    trainer = Trainer(model, optimizer, device=device)

    epochs = int(os.environ.get('EPOCHS', '1'))
    total_steps = len(dataloader) * epochs
    print(f'Starting training for {epochs} epoch(s)...\n')

    # Performance tracking
    epoch_metrics = []
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        avg_loss = trainer.train_epoch(dataloader, total_steps, epoch, scheduler=cosine_schedule)
        epoch_time = time.time() - epoch_start
        try:
            perplexity = math.exp(avg_loss)
        except OverflowError:
            perplexity = float('inf')
        print(f'\nEpoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
              f'Perplexity: {perplexity:.2f}, Time: {epoch_time:.1f}s')

        # Evaluate on the validation set after each epoch.
        val_loss = None
        val_perplexity = None
        if val_dataloader is not None:
            val_loss = trainer.evaluate(val_dataloader)
            try:
                val_perplexity = math.exp(val_loss)
            except OverflowError:
                val_perplexity = float('inf')
            print(f'Validation Loss: {val_loss:.4f}, '
                  f'Validation Perplexity: {val_perplexity:.2f}')

        epoch_metrics.append({
            'epoch': epoch + 1,
            'avg_loss': round(avg_loss, 6),
            'perplexity': round(perplexity, 4) if perplexity != float('inf') else None,
            'val_loss': round(val_loss, 6) if val_loss is not None else None,
            'val_perplexity': round(val_perplexity, 4) if val_perplexity not in (None, float('inf')) else None,
            'epoch_time_sec': round(epoch_time, 2),
        })

        model_path = os.path.join(root, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    total_time = time.time() - train_start

    # Save performance report
    performance = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'tokenizer': type(tokenizer).__name__,
        'num_tokens': len(tokens),
        'num_train_samples': len(dataset),
        'num_batches': len(dataloader),
        'total_parameters': total_params,
        'epochs': epochs,
        'total_train_time_sec': round(total_time, 2),
        'config': {
            'vocab_size': config.vocab_size,
            'block_size': config.block_size,
            'batch_size': config.batch_size,
            'n_layers': config.n_layers,
            'dim': config.dim,
            'n_heads': config.n_heads,
            'kv_heads': config.kv_heads,
            'dim_ff': config.dim_ff,
            'max_seq_len': config.max_seq_len,
        },
        'num_val_tokens': len(val_tokens) if val_tokens else 0,
        'num_val_samples': len(val_dataloader.dataset) if val_dataloader is not None else 0,
        'final_loss': epoch_metrics[-1]['avg_loss'] if epoch_metrics else None,
        'final_perplexity': epoch_metrics[-1]['perplexity'] if epoch_metrics else None,
        'final_val_loss': epoch_metrics[-1]['val_loss'] if epoch_metrics else None,
        'final_val_perplexity': epoch_metrics[-1]['val_perplexity'] if epoch_metrics else None,
        'epochs_detail': epoch_metrics,
    }

    perf_path = os.path.join(root, 'performance.json')
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(performance, f, indent=2)
    print(f'\nPerformance metrics saved to {perf_path}')

    print('\nTraining complete!')


if __name__ == '__main__':
    main()
