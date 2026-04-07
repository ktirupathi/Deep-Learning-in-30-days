"""Problem: Build robust train/val/test NLP input pipeline using HuggingFace datasets + PyTorch."""
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
raw = load_dataset("dbpedia_14")

# Create validation split from train
split = raw["train"].train_test_split(test_size=0.1, seed=42)

def encode(batch):
    x = tok(batch["content"], truncation=True, padding="max_length", max_length=128)
    x["labels"] = batch["label"]
    return x

train = split["train"].map(encode, batched=True)
valid = split["test"].map(encode, batched=True)
test = raw["test"].map(encode, batched=True)
for d in (train, valid, test):
    d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(valid, batch_size=64)
test_loader = DataLoader(test, batch_size=64)

print("pipeline ready", len(train_loader), len(val_loader), len(test_loader))
