"""
Example 10: Early stopping + checkpointing on AG News.
Dataset explanation:
- Uses real AG News train split with internal validation split.
Architecture explanation:
- Embedding mean-pooling classifier; focus is training strategy.
Training loop explanation:
- Monitor validation loss; save best model; stop after patience violations.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
raw = load_dataset("ag_news")
split = raw["train"].train_test_split(test_size=0.1, seed=7)

def encode(batch):
    x = tok(batch["text"], truncation=True, padding="max_length", max_length=96)
    x["labels"] = batch["label"]
    return x

train = split["train"].map(encode, batched=True)
val = split["test"].map(encode, batched=True)
for d in (train, val):
    d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=128)

class Net(nn.Module):
    def __init__(self, vocab):
        super().__init__(); self.e = nn.Embedding(vocab, 128); self.fc = nn.Linear(128, 4)
    def forward(self, ids, mask):
        pooled = (self.e(ids) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.fc(pooled)

model = Net(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

best_val = float("inf")
patience, bad_epochs = 2, 0
for epoch in range(10):
    model.train()
    for b in train_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        logits = model(b["input_ids"], b["attention_mask"])
        loss = ce(logits, b["labels"])
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); val_loss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            val_loss += float(ce(model(b["input_ids"], b["attention_mask"]), b["labels"]))
    val_loss /= len(val_loader)
    print(f"epoch={epoch} val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best_week1_agnews.pt")
        print("checkpoint saved")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Expected output: early stopping triggered")
            break

print("Improvements: restore best checkpoint before test eval; add LR scheduler and EMA.")
