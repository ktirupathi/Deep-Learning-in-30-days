"""
Example 06: GRU language model on WikiText-2.
Dataset explanation:
- WikiText-2 is a standard open benchmark for language modeling.
Architecture:
- Token embedding -> GRU -> vocab projection.
Math-to-code mapping:
- CE over shifted tokens approximates NLL; perplexity = exp(loss).
"""
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
text = "\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")["text"])
ids = tok(text, return_tensors="pt").input_ids[0][:22000]
seq = 64
X = ids.unfold(0, seq, seq)
Y = ids[1:].unfold(0, seq, seq)

class GRULM(nn.Module):
    def __init__(self, vocab, d=256, h=256):
        super().__init__()
        self.e = nn.Embedding(vocab, d)
        self.rnn = nn.GRU(d, h, batch_first=True)
        self.head = nn.Linear(h, vocab)
    def forward(self, x):
        o, _ = self.rnn(self.e(x))
        return self.head(o)

m = GRULM(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
ce = nn.CrossEntropyLoss()
for step in range(min(120, X.size(0)-1)):
    xb, yb = X[step:step+1].to(DEVICE), Y[step:step+1].to(DEVICE)
    logits = m(xb)
    loss = ce(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

print("Expected output: final loss", float(loss), "perplexity", float(torch.exp(loss.detach().cpu())))
print("Tensor shapes: logits=[1,T,V], labels=[1,T].")
print("Improvements: packed sequences, tied embeddings, dropout between recurrent layers.")
