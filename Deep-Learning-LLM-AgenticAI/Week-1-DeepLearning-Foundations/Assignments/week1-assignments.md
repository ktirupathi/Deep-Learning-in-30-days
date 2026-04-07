# Week 1 Assignments (Increasing Difficulty)

## Section A — Conceptual Questions (5)
1. Explain why embedding spaces enable semantic generalization while one-hot vectors do not.
2. Differentiate empirical risk minimization and true risk; why does generalization gap matter in NLP?
3. Why does gradient clipping help recurrent and long-context models?
4. Compare accuracy, macro-F1, and perplexity: when is each the right metric?
5. In production NLP, why is calibration and confidence analysis as important as raw accuracy?

## Section B — Mathematical Derivations (5)
1. Derive \(\partial\mathcal L/\partial z\) for softmax+cross-entropy from first principles.
2. Derive gradients for binary logistic regression with BCE: \(\nabla_w\mathcal J\) and \(\partial\mathcal J/\partial b\).
3. For mean-pooled embeddings + linear classifier, derive gradients with respect to embedding rows and classifier weights.
4. Expand chain rule through two-layer MLP with ReLU and CE loss; produce final \(\nabla_{W_1}\) expression.
5. Starting from autoregressive likelihood, derive perplexity and explain its relation to geometric mean probability.

## Section C — Coding Exercises (5)
1. Implement AG News classification with embedding mean pooling and report validation accuracy + macro-F1.
2. Replace pooling classifier with BiGRU encoder; compare parameter count and convergence speed.
3. Add mixed precision (`torch.cuda.amp`) and benchmark throughput vs baseline.
4. Implement error analysis script grouping failures by text-length buckets.
5. Build reproducibility pipeline: fixed seeds, config file, checkpoint resume, and JSON metrics logging.

## Section D — Mini Project (1)
### Title
**Intent Router v1 for Customer Support**

### Difficulty
Advanced beginner → intermediate.

### Problem statement
Build a multiclass intent classifier that routes user utterances to support teams.

### Dataset
Use `clinc_oos` (HuggingFace). Train on in-domain intents and evaluate robustness on out-of-scope examples.

### Required deliverables
- `train.py`: training with train/val split, early stopping, checkpointing
- `evaluate.py`: accuracy, macro-F1, confusion matrix
- `infer.py`: single and batch inference API
- `report.md`: equations, architecture choices, error categories, improvement ideas

### Stretch goals
- Distillation from a Transformer teacher
- Threshold-based OOS rejection
- Embedding visualization for intent clusters
