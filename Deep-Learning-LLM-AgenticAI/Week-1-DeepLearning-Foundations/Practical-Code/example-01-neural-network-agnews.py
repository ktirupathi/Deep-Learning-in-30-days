"""
Problem: Multiclass news topic classification on AG News.
Dataset: ag_news (HuggingFace datasets, real-world news corpus).
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH = 32
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
ds = load_dataset("ag_news")


def encode(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = batch["label"]
    return out

encoded = ds.map(encode, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(encoded["train"], batch_size=BATCH, shuffle=True)
test_loader = DataLoader(encoded["test"], batch_size=BATCH)


class AvgEmbedClassifier(nn.Module):
    def __init__(self, vocab_size, dim=256, num_labels=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)  # [B, T, D]
        mask = attention_mask.unsqueeze(-1)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(x)


model = AvgEmbedClassifier(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
acc = evaluate.load("accuracy")

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        opt.zero_grad()
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pred = model(batch["input_ids"], batch["attention_mask"]).argmax(-1)
        acc.add_batch(predictions=pred.cpu(), references=batch["labels"].cpu())

print("Test accuracy:", acc.compute())
