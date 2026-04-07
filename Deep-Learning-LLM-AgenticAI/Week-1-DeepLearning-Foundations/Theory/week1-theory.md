# Week 1 Theory — Deep Learning Foundations for LLM Engineers

## 1) What is Deep Learning in the LLM context?

### Layman explanation
Deep learning is a way to teach machines by example. Instead of manually writing all language rules, we feed text data to neural networks and let the model learn patterns (words, phrases, grammar, intent) by minimizing mistakes.

### Technical explanation
A deep neural network is a parameterized function
\[
f_\theta: \mathbb{R}^{n} \to \mathbb{R}^{m}
\]
trained by empirical risk minimization:
\[
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i)
\]
using first-order optimization (SGD/Adam) with gradients from backpropagation.

### Real-world use
- Intent classification for chatbots
- Toxicity/sentiment detection
- Next-token prediction (foundation of LLM pretraining)

### LLM relevance
LLMs are extremely large deep networks. Week 1 gives the same training foundations used later in Transformer pretraining and instruction tuning.

---

## 2) Representation learning for text

### Layman explanation
Words are converted into numbers (vectors). Similar words get similar vectors.

### Technical explanation
For token id \(t\), embedding lookup returns row \(E_t\) from matrix \(E\in \mathbb{R}^{V\times d}\). A sequence becomes \(X\in\mathbb{R}^{T\times d}\). This continuous representation enables differentiable learning.

### Real-world use
- Semantic retrieval
- Classification
- Prompt and context encoding

### LLM relevance
Token embeddings are the very first layer of every decoder-only and encoder-decoder LLM.

---

## 3) Forward pass, loss, and backprop

### Layman explanation
Model predicts, we compare against truth, compute error, then adjust model weights to reduce future error.

### Technical explanation
- Forward: \(z = W x + b\), \(a = \phi(z)\)
- Loss (classification): cross-entropy
- Backward: chain rule accumulates \(\partial \mathcal{L}/\partial \theta\)
- Update: \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}\)

### Real-world use
This loop is the engine of all training jobs.

### LLM relevance
Pretraining loss for causal LMs is cross-entropy over next-token distribution.

---

## 4) Optimization and stability

### Layman explanation
Learning rate controls how big each step is. Too big overshoots; too small learns slowly.

### Technical explanation
- SGD: \(\theta_{t+1}=\theta_t-\eta g_t\)
- Adam: adaptive moments \(m_t, v_t\) with bias correction
- Regularization: dropout, weight decay, early stopping

### Real-world use
Reduces training cost and improves generalization.

### LLM relevance
Modern LLM training heavily relies on AdamW + schedules + gradient clipping.

---

## 5) Sequence modeling before Transformers

### Layman explanation
RNN/LSTM/GRU process text token-by-token while carrying memory.

### Technical explanation
Hidden state recurrence:
\[
h_t = f(x_t, h_{t-1})
\]
LSTM gates improve gradient flow across long contexts.

### Real-world use
Small-footprint on-device NLP, baseline language modeling.

### LLM relevance
Understanding recurrence clarifies why self-attention replaced recurrence for long-range dependencies.

---

## 6) Evaluation culture

### Layman explanation
Accuracy alone can be misleading. Use several metrics and inspect mistakes.

### Technical explanation
- Classification: accuracy, F1, precision/recall
- Language modeling: perplexity
- Calibration/confusion matrix/error buckets

### Real-world use
Needed for production decisions and model governance.

### LLM relevance
Evaluation is central for alignment, retrieval quality, hallucination checks, and agent reliability.

---

## 7) Week 1 outcomes
By end of week, learners can:
1. Build text pipelines with real datasets.
2. Train PyTorch NLP models with proper loops.
3. Derive gradients for common layers/losses.
4. Evaluate and debug under/overfitting.
5. Prepare for Transformer internals in Week 2.
