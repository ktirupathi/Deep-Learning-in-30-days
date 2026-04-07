"""
Example 03: IMDB sentiment classification with embedding pooling.
Dataset explanation:
- IMDB is a real sentiment benchmark (binary labels).
Architecture:
- Embedding -> masked mean -> Linear(2 classes).
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
raw = load_dataset("imdb")

def encode(batch):
    x = tok(batch["text"], truncation=True, padding="max_length", max_length=256)
    x["labels"] = batch["label"]
    return x

enc = raw.map(encode, batched=True)
enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(enc["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(enc["test"], batch_size=32)

class SentimentNet(nn.Module):
    def __init__(self, vocab, d=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, 2)
    def forward(self, ids, mask):
        h = self.emb(ids)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.fc(pooled)

model = SentimentNet(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
ce = nn.CrossEntropyLoss()
metric = evaluate.load("f1")

for epoch in range(1):
    model.train()
    for b in train_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        logits = model(b["input_ids"], b["attention_mask"])
        loss = ce(logits, b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch={epoch} loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    for b in test_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        pred = model(b["input_ids"], b["attention_mask"]).argmax(-1)
        metric.add_batch(predictions=pred.cpu(), references=b["labels"].cpu())

print("Expected output: macro-F1 dictionary", metric.compute(average="macro"))
print("Improvements: BiGRU/Transformer encoder, focal loss for hard examples.")
