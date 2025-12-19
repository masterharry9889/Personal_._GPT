from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import pandas as pd

ds = pd.read_csv("D:/p/data/test.csv")

print(ds.head())

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

tokenizer.pre_tokenizer = None

tokenizer.decode = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size = 200_000,
    min_frequency = 2,
    show_progress = True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    special_tokens = [
        "<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>", "<|sep|>",
        "<|assistant|>", "<|user|>", "<|system|>"
    ]
)

tokenizer.train_from_iterator(ds['dialog'], trainer= trainer)

tokenizer.save("jarvis_tokenizer.json");