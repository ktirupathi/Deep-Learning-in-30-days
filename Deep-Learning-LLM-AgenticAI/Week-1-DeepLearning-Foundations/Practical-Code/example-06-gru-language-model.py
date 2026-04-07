"""Problem: Next-token language modeling with GRU on WikiText-2."""
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
text = "\n".join(ds["text"])
ids = tok(text, return_tensors="pt").input_ids[0][:20000]
seq = 64
X = ids.unfold(0, seq, seq)
Y = ids[1:].unfold(0, seq, seq)

class GRULM(nn.Module):
    def __init__(self,v,d=256,h=256):
        super().__init__(); self.e=nn.Embedding(v,d); self.gru=nn.GRU(d,h,batch_first=True); self.fc=nn.Linear(h,v)
    def forward(self,x):
        o,_=self.gru(self.e(x)); return self.fc(o)

m = GRULM(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
ce = nn.CrossEntropyLoss()
for i in range(min(100, X.size(0)-1)):
    xb = X[i:i+1].to(DEVICE)
    yb = Y[i:i+1].to(DEVICE)
    logits = m(xb)
    loss = ce(logits.view(-1, logits.size(-1)), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print("final_loss", float(loss), "perplexity", float(torch.exp(loss.detach().cpu())))
