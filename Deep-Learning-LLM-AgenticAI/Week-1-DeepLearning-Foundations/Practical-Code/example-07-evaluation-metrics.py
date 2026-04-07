"""
Example 07: Metric toolkit (accuracy, precision, recall, macro-F1).
Dataset explanation:
- AG News test subset demonstrates multiclass evaluation.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = load_dataset("ag_news", split="test[:2000]")

def prep(b):
    x = tok(b["text"], truncation=True, padding="max_length", max_length=128)
    x["labels"] = b["label"]
    return x

ds = ds.map(prep, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=64)

class Tiny(nn.Module):
    def __init__(self, vocab):
        super().__init__(); self.e = nn.Embedding(vocab, 32); self.fc = nn.Linear(32, 4)
    def forward(self, ids, mask):
        pooled = (self.e(ids) * mask.unsqueeze(-1)).mean(1)
        return self.fc(pooled)

m = Tiny(tok.vocab_size)
acc, prec, rec, f1 = evaluate.load("accuracy"), evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1")
with torch.no_grad():
    for b in loader:
        p = m(b["input_ids"], b["attention_mask"]).argmax(-1)
        y = b["labels"]
        for metric in (acc, prec, rec, f1):
            metric.add_batch(predictions=p, references=y)

print("Expected output:")
print("accuracy", acc.compute())
print("precision_macro", prec.compute(average="macro"))
print("recall_macro", rec.compute(average="macro"))
print("f1_macro", f1.compute(average="macro"))
print("Improvements: confusion matrix, calibration curves, class-wise error slices.")
