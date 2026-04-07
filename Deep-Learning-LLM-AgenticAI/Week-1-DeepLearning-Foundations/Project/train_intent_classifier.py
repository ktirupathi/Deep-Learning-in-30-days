"""Week-1 project implementation: CLINC OOS intent classifier with BiGRU in PyTorch."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

MODEL = "distilbert-base-uncased"
MAX_LEN = 48
BATCH = 64
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
raw = load_dataset("clinc_oos", "plus")

# Filter to in-domain intents for standard multiclass classification
train_raw = raw["train"]
val_raw = raw["validation"]
test_raw = raw["test"]

num_labels = len(set(train_raw["intent"]))

def encode(batch):
    z = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    z["labels"] = batch["intent"]
    return z

train = train_raw.map(encode, batched=True)
val = val_raw.map(encode, batched=True)
test = test_raw.map(encode, batched=True)
for d in (train, val, test):
    d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH)
test_loader = DataLoader(test, batch_size=BATCH)


class BiGRUIntent(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, labels=150):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, labels)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        h, _ = self.gru(x)
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled)


model = BiGRUIntent(tok.vocab_size, labels=num_labels).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

best_val = 1e9
for epoch in range(EPOCHS):
    model.train()
    for b in train_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        logits = model(b["input_ids"], b["attention_mask"])
        loss = criterion(logits, b["labels"])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            logits = model(b["input_ids"], b["attention_mask"])
            val_loss += float(criterion(logits, b["labels"]))
    val_loss /= len(val_loader)
    print(f"epoch={epoch} val_loss={val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "week1_intent_best.pt")

model.load_state_dict(torch.load("week1_intent_best.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    for b in test_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        p = model(b["input_ids"], b["attention_mask"]).argmax(-1)
        y = b["labels"]
        acc.add_batch(predictions=p.cpu(), references=y.cpu())
        f1.add_batch(predictions=p.cpu(), references=y.cpu())

print("test_accuracy", acc.compute())
print("test_macro_f1", f1.compute(average="macro"))
