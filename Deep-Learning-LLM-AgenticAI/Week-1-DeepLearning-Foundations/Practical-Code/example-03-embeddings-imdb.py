"""Problem: Binary sentiment classification with embedding + mean pooling on IMDB dataset."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = load_dataset("imdb")

def prep(b):
    x = tok(b["text"], truncation=True, padding="max_length", max_length=256)
    x["labels"] = b["label"]
    return x

ds = ds.map(prep, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(ds["train"], batch_size=16, shuffle=True)
test_loader = DataLoader(ds["test"], batch_size=32)

class Model(nn.Module):
    def __init__(self, v, d=128):
        super().__init__()
        self.e = nn.Embedding(v, d)
        self.h = nn.Linear(d, 2)
    def forward(self, ids, mask):
        x = self.e(ids)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.h(x)

m = Model(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
ce = nn.CrossEntropyLoss()
metric = evaluate.load("f1")

for _ in range(1):
    m.train()
    for b in train_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        loss = ce(m(b["input_ids"], b["attention_mask"]), b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()

m.eval()
with torch.no_grad():
    for b in test_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        p = m(b["input_ids"], b["attention_mask"]).argmax(-1)
        metric.add_batch(predictions=p.cpu(), references=b["labels"].cpu())
print(metric.compute(average="macro"))
