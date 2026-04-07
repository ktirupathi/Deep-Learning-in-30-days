# Week 1 Project — Intent Classification System (Production-Style Starter)

## 1) Problem statement
Build an intent classification model that routes user utterances to the correct support intent class. This simulates the first stage of many agentic AI systems: deciding *which tool, policy, or sub-agent* should handle a user message.

## 2) Dataset explanation
- Dataset: `clinc_oos` (HuggingFace).
- Contains in-domain intent classes and out-of-scope examples.
- Why realistic: short, user-generated utterances with lexical variability and ambiguous phrasing.

## 3) Mathematical formulation
Given utterance tokens \(x_{1:T}\), predict class \(y\in\{1,\dots,C\}\):
\[
\hat y = \arg\max_c\; p(c\mid x_{1:T};\theta)
\]
Training objective:
\[
\min_\theta \; -\frac1N\sum_{i=1}^{N} \log p(y_i\mid x_i;\theta)
\]

## 4) Architecture diagram (text)
```
Input text
  -> Tokenizer (DistilBERT tokenizer)
  -> Token IDs [B,T]
  -> Embedding layer [B,T,D]
  -> BiGRU encoder [B,T,2H]
  -> Masked mean pooling [B,2H]
  -> Linear classifier [B,C]
  -> Softmax probabilities
```

## 5) Training strategy
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Gradient clipping: max-norm 1.0
- Early model selection using validation loss checkpointing
- Evaluation on held-out test split

## 6) Evaluation metrics
- Accuracy
- Macro-F1 (important for class imbalance)
- Optional confusion matrix export

## 7) Full code explanation
See `train_intent_classifier.py` for complete pipeline:
- dataset loading and tokenization,
- DataLoader creation,
- model definition,
- training/validation loops,
- checkpoint save/load,
- final test metrics.

## 8) Improvements
- Add OOS rejection threshold from softmax confidence.
- Replace BiGRU encoder with pretrained Transformer encoder.
- Add calibration layer for confidence-aware routing.
- Add error slices by intent frequency and utterance length.
