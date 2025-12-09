import torch
import torch.utils.data as DataLoader
from tokenizer import Tokenizer
from model_f.model import Model
from model_f.config import Config
from training import trainer
from training.dataset import CausalTextDataset
from training.optimizer import get_optimizer
from training.schedules import cosine_schedule
from training.trainer import Trainer


# Load Tokenizer
tokenizer = Tokenizer.from_file("tokenizer\jarvis_tokenizer.json")

# Load and Tokenize dataset
with open("data/train.txt", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text).ids

dataset = CausalTextDataset(tokens, block_size=Config.block_size)
dataloader = DataLoader.DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

# Initialize Model
model = Model(Config)
optimizer = get_optimizer(model)
trainer = Trainer(model, optimizer)

# Train Model
epochs = 5
total_steps = len(dataloader) * epoch

# Training Loop
for epoch in range(epochs):
    avg_loass = trainer.train_epoch(dataloader, total_steps, epoch, scheduler=cosine_schedule)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loass:.4f}")

    # Save Model
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    print(f"Model saved to model_epoch_{epoch+1}.pth")
