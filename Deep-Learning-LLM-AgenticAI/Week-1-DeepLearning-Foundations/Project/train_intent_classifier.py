"""
Week-1 Project: CLINC OOS intent classifier with BiGRU.

Dataset explanation:
- CLINC OOS is a real intent dataset with many intents and out-of-scope utterances.
- Here we train multiclass intent prediction on in-domain split keys.

Architecture explanation:
- Embedding -> BiGRU -> masked mean pooling -> linear classifier.

Math-to-code mapping:
- pooled = sum(mask * h_t) / sum(mask)
- logits = W pooled + b
- loss = CrossEntropy(logits, labels)
"""
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

# 1) Load tokenizer + dataset
# Expected keys in clinc_oos/plus: train, validation, test
# label column for intents: "intent"
tok = AutoTokenizer.from_pretrained(MODEL)
raw = load_dataset("clinc_oos", "plus")
train_raw, val_raw, test_raw = raw["train"], raw["validation"], raw["test"]
num_labels = len(set(train_raw["intent"]))

# 2) Preprocessing
# Shapes after set_format:
# input_ids [B,T], attention_mask [B,T], labels [B]
def preprocess(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = batch["intent"]
    return out

train = train_raw.map(preprocess, batched=True)
val = val_raw.map(preprocess, batched=True)
test = test_raw.map(preprocess, batched=True)
for d in (train, val, test):
    d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH)
test_loader = DataLoader(test, batch_size=BATCH)


class BiGRUIntent(nn.Module):
    """Embedding + bidirectional GRU sentence encoder + linear head."""
    def __init__(self, vocab_size, emb_dim=128, hidden=128, labels=150):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, labels)

    def forward(self, input_ids, attention_mask):
        # input_ids: [B,T]
        emb = self.emb(input_ids)                 # [B,T,D]
        h, _ = self.gru(emb)                      # [B,T,2H]
        mask = attention_mask.unsqueeze(-1)       # [B,T,1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B,2H]
        logits = self.fc(pooled)                  # [B,C]
        return logits


model = BiGRUIntent(tok.vocab_size, labels=num_labels).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# 3) Training strategy: validation checkpointing
best_val = float("inf")
for epoch in range(EPOCHS):
    model.train()
    for b in train_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        logits = model(b["input_ids"], b["attention_mask"])
        loss = criterion(logits, b["labels"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            val_loss += float(criterion(model(b["input_ids"], b["attention_mask"]), b["labels"]))
    val_loss /= len(val_loader)
    print(f"epoch={epoch} val_loss={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "week1_intent_best.pt")

# 4) Evaluation
model.load_state_dict(torch.load("week1_intent_best.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    for b in test_loader:
        b = {k: v.to(DEVICE) for k, v in b.items()}
        pred = model(b["input_ids"], b["attention_mask"]).argmax(-1)
        y = b["labels"]
        acc_metric.add_batch(predictions=pred.cpu(), references=y.cpu())
        f1_metric.add_batch(predictions=pred.cpu(), references=y.cpu())

print("Expected output: test accuracy", acc_metric.compute())
print("Expected output: test macro-F1", f1_metric.compute(average="macro"))
print("Improvements: OOS thresholding, calibration, pretrained encoder fine-tuning.")
