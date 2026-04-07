"""
Example 04: Optimizer comparison (SGD vs AdamW) on AG News subset.
Dataset explanation:
- AG News train subset enables quick iteration while still real-world.
Architecture:
- Same network for fair optimizer comparison.
"""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = load_dataset("ag_news", split="train[:4000]")

def prep(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=96)
    z["labels"] = b["label"]
    return z

ds = ds.map(prep, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self, vocab):
        super().__init__(); self.e = nn.Embedding(vocab, 64); self.fc = nn.Linear(64, 4)
    def forward(self, ids, mask):
        x = self.e(ids)
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.fc(x)

for name, opt_ctor in {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.05),
    "AdamW": lambda p: torch.optim.AdamW(p, lr=1e-3)
}.items():
    model = Net(tok.vocab_size).to(DEVICE)
    opt = opt_ctor(model.parameters())
    ce = nn.CrossEntropyLoss()
    model.train()
    for step, b in enumerate(loader):
        if step >= 80: break
        b = {k: v.to(DEVICE) for k, v in b.items()}
        loss = ce(model(b["input_ids"], b["attention_mask"]), b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(name, "expected output: final loss", float(loss))

print("Improvements: tune LR schedule, warmup, weight decay sweep, gradient norm logging.")
