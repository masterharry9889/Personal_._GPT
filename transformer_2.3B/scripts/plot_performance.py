import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend so it works headless.
import matplotlib.pyplot as plt


def load_performance(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_performance(perf, out_path):
    epochs_detail = perf.get('epochs_detail', [])
    if not epochs_detail:
        raise ValueError('No epochs_detail found in performance.json; nothing to plot.')

    epochs = [e['epoch'] for e in epochs_detail]
    train_loss = [e.get('avg_loss') for e in epochs_detail]
    val_loss = [e.get('val_loss') for e in epochs_detail]
    train_ppl = [e.get('perplexity') for e in epochs_detail]
    val_ppl = [e.get('val_perplexity') for e in epochs_detail]

    has_val_loss = any(v is not None for v in val_loss)
    has_val_ppl = any(v is not None for v in val_ppl)

    # With a single epoch a line is invisible, so use markers + lines.
    marker = 'o' if len(epochs) == 1 else 'o-'

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Loss curve ---
    ax_loss.plot(epochs, train_loss, marker, color='#1f77b4', label='Train loss')
    if has_val_loss:
        ax_loss.plot(epochs, val_loss, marker, color='#ff7f0e', label='Validation loss')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    if len(epochs) == 1:
        ax_loss.set_xticks(epochs)

    # --- Perplexity curve ---
    ax_ppl.plot(epochs, train_ppl, marker, color='#1f77b4', label='Train perplexity')
    if has_val_ppl:
        ax_ppl.plot(epochs, val_ppl, marker, color='#ff7f0e', label='Validation perplexity')
    ax_ppl.set_title('Perplexity')
    ax_ppl.set_xlabel('Epoch')
    ax_ppl.set_ylabel('Perplexity')
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.legend()
    if len(epochs) == 1:
        ax_ppl.set_xticks(epochs)

    tokenizer = perf.get('tokenizer', 'unknown')
    params = perf.get('total_parameters')
    subtitle = f'Tokenizer: {tokenizer}'
    if params is not None:
        subtitle += f'  |  Parameters: {params:,}'
    fig.suptitle(f'Transformer Training Performance\n{subtitle}', fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f'Figure saved to {out_path}')


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    perf_path = os.path.join(root, 'performance.json')
    out_path = os.path.join(root, 'performance.png')

    if not os.path.exists(perf_path):
        print(f'performance.json not found at {perf_path}. Run training first.')
        sys.exit(1)

    perf = load_performance(perf_path)
    plot_performance(perf, out_path)


if __name__ == '__main__':
    main()
