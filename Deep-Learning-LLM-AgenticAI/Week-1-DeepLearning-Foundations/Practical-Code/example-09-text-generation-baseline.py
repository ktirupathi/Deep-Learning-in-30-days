"""Problem: Train tiny next-token MLP baseline on WikiText-2 for language modeling intuition."""
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")["text"])
ids = tok(text, return_tensors="pt").input_ids[0][:12000]
ctx = 16
X, Y = [], []
for i in range(len(ids)-ctx-1):
    X.append(ids[i:i+ctx]); Y.append(ids[i+ctx])
X, Y = torch.stack(X[:5000]), torch.tensor(Y[:5000])

class MLPNextToken(nn.Module):
    def __init__(self,v,d=64,ctx=16):
        super().__init__(); self.e=nn.Embedding(v,d); self.fc=nn.Sequential(nn.Linear(d*ctx,256), nn.ReLU(), nn.Linear(256,v))
    def forward(self,x):
        return self.fc(self.e(x).reshape(x.size(0), -1))

m = MLPNextToken(tok.vocab_size, ctx=ctx).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
for step in range(200):
    idx = torch.randint(0, X.size(0), (64,))
    xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
    loss = ce(m(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
print("loss", float(loss), "ppl", float(torch.exp(loss.detach().cpu())))
