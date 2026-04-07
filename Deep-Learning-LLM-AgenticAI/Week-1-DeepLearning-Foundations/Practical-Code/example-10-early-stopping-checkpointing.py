"""Problem: Add early stopping and checkpointing to real text classification training (AG News)."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
raw = load_dataset("ag_news")
split = raw["train"].train_test_split(test_size=0.1, seed=7)

def enc(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=96)
    z["labels"] = b["label"]
    return z

train = split["train"].map(enc, batched=True)
val = split["test"].map(enc, batched=True)
for d in (train, val):
    d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=128)

class Net(nn.Module):
    def __init__(self,v):
        super().__init__(); self.e=nn.Embedding(v,128); self.fc=nn.Linear(128,4)
    def forward(self,i,m):
        x=(self.e(i)*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp(min=1); return self.fc(x)

m = Net(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
best = 1e9; patience = 2; bad = 0
for epoch in range(10):
    m.train()
    for b in train_loader:
        b = {k:v.to(DEVICE) for k,v in b.items()}
        loss = ce(m(b["input_ids"], b["attention_mask"]), b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); vloss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b = {k:v.to(DEVICE) for k,v in b.items()}
            vloss += float(ce(m(b["input_ids"], b["attention_mask"]), b["labels"]))
    vloss /= len(val_loader)
    print("epoch", epoch, "val_loss", vloss)
    if vloss < best:
        best = vloss; bad = 0; torch.save(m.state_dict(), "best_week1_agnews.pt")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping triggered")
            break
