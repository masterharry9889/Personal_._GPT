import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader, total_steps, epoch, scheduler=None):
        self.model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(tqdm(dataloader)):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if scheduler:
                scheduler(self.optimizer, step + epoch * len(dataloader), total_steps)

            running_loss += loss.item()

        return running_loss / max(1, len(dataloader))

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        for x, y in tqdm(dataloader, desc='eval'):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            running_loss += loss.item()

        return running_loss / max(1, len(dataloader))