"""
Example 09: Tiny context-window MLP next-token model.
Dataset explanation:
- WikiText-2 text converted into context-target pairs.
Architecture:
- Embedding(context) -> flatten -> MLP -> vocab logits.
"""
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
for i in range(len(ids) - ctx - 1):
    X.append(ids[i:i+ctx]); Y.append(ids[i+ctx])
X = torch.stack(X[:5000]); Y = torch.tensor(Y[:5000])

class MLPNext(nn.Module):
    def __init__(self, vocab, d=64, ctx=16):
        super().__init__()
        self.e = nn.Embedding(vocab, d)
        self.net = nn.Sequential(nn.Linear(d * ctx, 256), nn.ReLU(), nn.Linear(256, vocab))
    def forward(self, x):
        return self.net(self.e(x).reshape(x.size(0), -1))

m = MLPNext(tok.vocab_size, ctx=ctx).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
for step in range(220):
    idx = torch.randint(0, X.size(0), (64,))
    xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
    logits = m(xb)
    loss = ce(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()

print("Expected output: loss and perplexity", float(loss), float(torch.exp(loss.detach().cpu())))
print("Shapes: xb=[B,ctx], logits=[B,V], labels=[B]")
print("Improvements: causal self-attention, tied embeddings, sampled softmax.")
