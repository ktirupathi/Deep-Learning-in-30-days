"""Problem: Compute precision/recall/F1/accuracy for multiclass text task (AG News)."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = load_dataset("ag_news", split="test[:2000]")

def prep(b):
    z = tok(b["text"], truncation=True, padding="max_length", max_length=128)
    z["labels"] = b["label"]
    return z

ds = ds.map(prep, batched=True)
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(ds, batch_size=64)

class Dummy(nn.Module):
    def __init__(self,v):
        super().__init__(); self.e=nn.Embedding(v,32); self.f=nn.Linear(32,4)
    def forward(self,i,m):
        return self.f((self.e(i)*m.unsqueeze(-1)).mean(1))

m = Dummy(tok.vocab_size)
acc = evaluate.load("accuracy"); pr = evaluate.load("precision"); rc = evaluate.load("recall"); f1 = evaluate.load("f1")
with torch.no_grad():
    for b in loader:
        p = m(b["input_ids"], b["attention_mask"]).argmax(-1)
        y = b["labels"]
        acc.add_batch(predictions=p, references=y)
        pr.add_batch(predictions=p, references=y)
        rc.add_batch(predictions=p, references=y)
        f1.add_batch(predictions=p, references=y)
print("acc", acc.compute())
print("precision_macro", pr.compute(average="macro"))
print("recall_macro", rc.compute(average="macro"))
print("f1_macro", f1.compute(average="macro"))
