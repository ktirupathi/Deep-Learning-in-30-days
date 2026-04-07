"""
Example 01: AG News topic classification with embedding mean pooling.
Dataset explanation:
- AG News has 4 classes: World, Sports, Business, Sci/Tech.
- Real-world usefulness: baseline news router and moderation pipelines.

Architecture explanation:
- Token IDs -> Embedding[V, D] -> Masked mean pooling -> Linear(D -> 4).
- This is a strong baseline before moving to Transformers.

Tensor shapes:
- input_ids: [B, T]
- embeddings: [B, T, D]
- pooled: [B, D]
- logits: [B, 4]

Math-to-code mapping:
- s = sum(m_i * e_i) / sum(m_i)
- z = W s + b
- L = CrossEntropy(z, y)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "distilbert-base-uncased"
MAX_LEN, BATCH, EPOCHS = 128, 32, 2

tok = AutoTokenizer.from_pretrained(MODEL)
raw = load_dataset("ag_news")

def preprocess(batch):
    out = tok(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = batch["label"]
    return out

enc = raw.map(preprocess, batched=True)
enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_loader = DataLoader(enc["train"], batch_size=BATCH, shuffle=True)
test_loader = DataLoader(enc["test"], batch_size=BATCH)

class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, d=256, classes=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.head = nn.Linear(d, classes)

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

model = MeanPoolClassifier(tok.vocab_size).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()
metric = evaluate.load("accuracy")

# Training loop explanation:
# 1) forward -> loss
# 2) zero_grad -> backward -> step
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch={epoch} train_loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        pred = model(batch["input_ids"], batch["attention_mask"]).argmax(-1)
        metric.add_batch(predictions=pred.cpu(), references=batch["labels"].cpu())

print("Expected output: accuracy dictionary", metric.compute())
print("Improvements: pretrained encoder, label smoothing, scheduler, AMP.")
