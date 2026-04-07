"""Problem: Measure impact of dropout on Yelp polarity generalization."""
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_dataset("yelp_polarity", split="train[:10000]")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def map_fn(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=128)
    z["labels"] = b["label"]
    return z

ds = ds.map(map_fn, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=32, shuffle=True)

class DropoutCls(nn.Module):
    def __init__(self,v):
        super().__init__(); self.e=nn.Embedding(v,128); self.drop=nn.Dropout(0.5); self.fc=nn.Linear(128,2)
    def forward(self,i,m):
        x=self.e(i); x=(x*m.unsqueeze(-1)).mean(1); return self.fc(self.drop(x))

m = DropoutCls(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
for epoch in range(2):
    m.train(); epoch_loss=0
    for b in loader:
        b = {k:v.to(DEVICE) for k,v in b.items()}
        loss = ce(m(b["input_ids"], b["attention_mask"]), b["labels"])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        epoch_loss += float(loss)
    print("epoch", epoch, "loss", epoch_loss/len(loader))
