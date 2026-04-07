# Week 1 — Deep Learning Foundations for NLP, LLM Pretraining, and Agentic AI

## Week Goal
This week transforms learners from “I can run a notebook” to “I understand why the model learns.” We focus on foundational deep learning mechanisms that later become Transformer internals, LLM pretraining objectives, and agentic model control loops.

## Day-by-Day Learning Path (Days 1–7)
1. **Day 1 — Tensor and gradient fluency**: vectorization, broadcasting, automatic differentiation, and numerical stability.
2. **Day 2 — Learning as optimization**: loss landscapes, gradient descent variants, initialization, and regularization.
3. **Day 3 — Text representation**: tokenization, vocabulary construction, embeddings, sequence padding/masking.
4. **Day 4 — Supervised NLP baselines**: embedding averaging and MLP/linear heads for robust baselines.
5. **Day 5 — Sequence modeling**: RNN/GRU language-modeling and sequence classification fundamentals.
6. **Day 6 — Evaluation and error analysis**: metrics, confusion analysis, calibration, and failure-pattern audits.
7. **Day 7 — Integrated weekly project**: production-style intent classifier with reproducibility and checkpointing.

## Folder Expectations
- **Theory/**: each core concept is explained for beginners and technical interviews.
- **Maths/**: full derivations from scalar equations to matrix calculus and backprop.
- **Practical-Code/**: 10 worked examples with real datasets and explicit math-to-code mapping.
- **Assignments/**: increasing-difficulty conceptual, derivation, and coding tasks.
- **Project/**: full project package with architecture text-diagram, formulas, and implementation.

## Tools and Libraries
- `torch`, `torch.nn`, `torch.optim`
- HuggingFace `datasets` and `transformers` tokenizers
- `evaluate` for metrics

## Deliverables after Week 1
By the end of this week, learners can:
- derive cross-entropy gradients without skipping steps,
- implement robust data pipelines for real text datasets,
- train and evaluate baseline NLP models in PyTorch,
- explain implementation choices in interview-ready language,
- connect foundational math to LLM training dynamics.
