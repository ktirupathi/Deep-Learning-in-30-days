"""Problem: Compare SGD vs AdamW on AG News text classifier."""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = load_dataset("ag_news", split="train[:5000]")

def f(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=96)
    z["labels"] = b["label"]
    return z

ds = ds.map(f, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self, v):
        super().__init__(); self.e=nn.Embedding(v,64); self.fc=nn.Linear(64,4)
    def forward(self,i,m):
        x=self.e(i); x=(x*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp(min=1); return self.fc(x)

base = Net(tok.vocab_size).to(DEVICE)
models = {
    "SGD": (copy.deepcopy(base), torch.optim.SGD(copy.deepcopy(base).parameters(), lr=0.1)),
    "AdamW": (copy.deepcopy(base), torch.optim.AdamW(copy.deepcopy(base).parameters(), lr=1e-3)),
}
ce = nn.CrossEntropyLoss()

for name, (model, opt) in models.items():
    model.to(DEVICE).train()
    for step, b in enumerate(loader):
        if step >= 100: break
        b = {k: v.to(DEVICE) for k, v in b.items()}
        loss = ce(model(b["input_ids"], b["attention_mask"]), b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(name, "last_loss", float(loss))
