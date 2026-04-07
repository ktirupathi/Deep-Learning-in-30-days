# Week 1 Assignments

## A) Conceptual Questions (5)
1. Why are embeddings superior to one-hot vectors for semantic generalization?
2. Explain underfitting vs overfitting in the context of NLP classification.
3. Why does cross-entropy pair naturally with softmax for multiclass tasks?
4. Compare SGD and Adam in terms of convergence behavior and memory.
5. Why is perplexity preferred over raw loss for language modeling interpretation?

## B) Mathematical Problems (5)
1. Derive \(\partial \mathcal{L}/\partial z\) for softmax + cross-entropy.
2. For \(y=\sigma(wx+b)\) with BCE loss, derive gradients wrt \(w,b\).
3. Given a 2-layer MLP with ReLU, derive \(\partial \mathcal{L}/\partial W_1\).
4. Show how L2 regularization modifies gradient update rules.
5. Compute perplexity from token log-likelihoods \([-1.2,-0.8,-1.5,-0.5]\).

## C) Coding Assignments (5)
1. Train an AG News classifier using averaged token embeddings + linear head.
2. Reproduce the same task with a GRU and compare validation accuracy.
3. Build a reusable training loop with gradient clipping + early stopping.
4. Implement error analysis that prints the top-20 misclassified samples.
5. Implement an experiment logger saving config, metrics, and model checkpoint.

## D) Mini Project (1)
### Title
**Customer Support Intent Router**

### Objective
Build a 5-class intent classifier trained on the CLINC OOS dataset (subset allowed).

### Requirements
- PyTorch implementation with train/validation/test split
- HuggingFace tokenizer + dataset pipeline
- Metrics: accuracy, macro-F1, confusion matrix
- Ablation: embedding-only vs GRU model
- Report: architecture, math, training curves, and failure cases

### Deliverables
- `train.py`, `evaluate.py`, `inference.py`
- `report.md` with equations and results
- `requirements.txt`
