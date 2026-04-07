"""
Example 08: Industrial-style dataset pipeline for DBPedia.
Dataset explanation:
- DBPedia 14-class ontology dataset; realistic multiclass text benchmark.
Architecture explanation:
- This script builds data loaders, not model training.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
raw = load_dataset("dbpedia_14")
split = raw["train"].train_test_split(test_size=0.1, seed=42)

def encode(batch):
    x = tok(batch["content"], truncation=True, padding="max_length", max_length=128)
    x["labels"] = batch["label"]
    return x

train = split["train"].map(encode, batched=True)
val = split["test"].map(encode, batched=True)
test = raw["test"].map(encode, batched=True)
for part in (train, val, test):
    part.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

print("Expected output: data loaders ready")
print("num_batches", len(train_loader), len(val_loader), len(test_loader))
print("Tensor shapes from a batch: input_ids [B,T], attention_mask [B,T], labels [B]")
print("Improvements: dynamic padding via DataCollatorWithPadding; streaming mode for huge corpora.")
