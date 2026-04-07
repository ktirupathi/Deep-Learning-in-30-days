"""
Example 05: Regularization with dropout on Yelp polarity.
Dataset explanation:
- Yelp polarity contains long-form sentiment text with real lexical noise.
Architecture:
- Embedding -> dropout -> linear classifier.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = load_dataset("yelp_polarity", split="train[:12000]")

def prep(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=128)
    z["labels"] = b["label"]
    return z

ds = ds.map(prep, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=32, shuffle=True)

class DropCls(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.e = nn.Embedding(vocab, 128)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 2)
    def forward(self, ids, mask):
        h = self.e(ids)
        pooled = (h * mask.unsqueeze(-1)).mean(1)
        return self.fc(self.drop(pooled))

m = DropCls(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()
for epoch in range(2):
    m.train(); epoch_loss = 0.0
    for b in loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        logits = m(b["input_ids"], b["attention_mask"])
        loss = ce(logits, b["labels"])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        epoch_loss += float(loss)
    print(f"epoch={epoch} expected output: avg_loss={epoch_loss/len(loader):.4f}")

print("Shape notes: logits=[B,2], labels=[B].")
print("Improvements: compare weight decay vs dropout-only; add validation split.")
