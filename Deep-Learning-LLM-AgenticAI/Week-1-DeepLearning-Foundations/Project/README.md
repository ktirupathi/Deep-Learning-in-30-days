# Week 1 Project — Intent Classification Service

## Project Goal
Train and evaluate a robust text intent classifier using a real dataset so learners practice full ML workflow before moving to Transformers.

## Dataset
- **Dataset**: `clinc_oos` (HuggingFace datasets)
- **Task**: intent classification
- **Input**: user utterance text
- **Output**: intent class id

## Architecture
- Tokenizer: `distilbert-base-uncased` tokenizer
- Model: Embedding → BiGRU → Mean Pool → Linear classifier
- Loss: CrossEntropyLoss
- Optimizer: AdamW

## Deliverables
- `train_intent_classifier.py`
- `architecture.md`
- `math-explanation.md`

## Evaluation
- Accuracy
- Macro-F1
- Confusion matrix export
