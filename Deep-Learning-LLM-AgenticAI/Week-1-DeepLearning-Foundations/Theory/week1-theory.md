# Week 1 Theory — Deep Learning Foundations for LLM Engineers

## Concept 1 — Representation Learning and Embeddings

### Layman explanation
A language model cannot reason directly over words like humans do; it needs numbers. But if we assign arbitrary IDs to words (for example, `cat=7`, `dog=52`, `rocket=903`), the model still cannot know that “cat” and “dog” are related while “rocket” is different. Embeddings solve this by giving each token a dense vector of learned features. You can imagine every word getting a coordinate in a semantic map. Words used in similar contexts are pulled closer together during training. Over time, the model builds a geometric understanding of language: professions cluster together, emotions cluster, locations cluster, etc.

A beginner-friendly way to see this: embeddings are like profile cards for words. A card might encode hidden properties such as “is food-related,” “is formal language,” or “is often used in financial news.” We do not hard-code those features. The network discovers them from data by minimizing task loss.

### Technical explanation
For vocabulary size \(V\) and embedding width \(d\), define trainable matrix \(E \in \mathbb{R}^{V\times d}\). Given token id \(t\), embedding lookup outputs row vector \(e_t = E[t]\in\mathbb{R}^{d}\). For sequence \((t_1,\ldots,t_T)\), we obtain \(X\in\mathbb{R}^{T\times d}\). This operation is equivalent to multiplying one-hot vectors by \(E\), but implemented efficiently as indexing.

In sentence classification baselines, we often pool token vectors into one vector:
\[
s = \frac{\sum_{i=1}^{T} m_i x_i}{\sum_{i=1}^{T}m_i}
\]
where \(m_i\in\{0,1\}\) masks padding. Then classifier logits are \(z = Ws+b\). Backpropagation sends gradients from the classifier through pooled representation back to each token embedding row. Only vocabulary rows appearing in a batch get non-zero updates, which is computationally efficient.

### Why needed for LLMs
Every modern LLM begins with token embeddings (plus positional information). Prompt interpretation, retrieval context encoding, instruction following, and generation quality all depend on representational quality at this first layer. In decoder-only models, embedding matrix is often tied with output projection, meaning one parameter matrix influences both reading and generation. Embedding quality affects long-tail vocabulary handling, multilingual transfer, and factual grounding behavior.

### Real-world analogy
Think of a large university campus map. Every building (word) has coordinates. Buildings with similar function (labs, dorms, libraries) are located in meaningful neighborhoods. If we move one building, pathways to many others change. Similarly, when training updates a token vector, relative geometry changes and influences model decisions everywhere that token participates.

### Common mistakes
1. Treating token IDs as meaningful numbers (ordinal misconception).
2. Ignoring padding mask during pooling, which corrupts sentence vectors.
3. Using extremely small embedding dimensions for semantically rich tasks.
4. Failing to handle unknown or rare tokens consistently between train and inference.
5. Assuming embeddings are static; they should be monitored and sometimes fine-tuned.
6. Not checking tokenization drift across libraries/models.

### Interview notes
- Explain why one-hot vectors are sparse and cannot express similarity directly.
- Mention subword tokenization (BPE/WordPiece) and its role in open-vocabulary robustness.
- Discuss embedding tying and parameter efficiency in LLMs.
- Be ready to explain why masked pooling is needed for padded sequences.

---

## Concept 2 — Loss, Backpropagation, and Optimization

### Layman explanation
Training a deep model is an error-correction cycle. The model predicts, we measure how wrong it is, then we adjust internal knobs (weights) so next prediction is better. Backpropagation is the mechanism that distributes blame: it tells each knob whether it made the final mistake larger or smaller. Optimization algorithms then decide how big each adjustment should be.

Suppose a chatbot misclassifies a user request as “billing” instead of “technical support.” The loss increases. Backprop traces this error backward through classifier layer, sequence encoder, and embeddings, updating all relevant parameters. Repeating this on millions of examples gradually improves behavior.

### Technical explanation
Let \(\hat y=f_\theta(x)\), objective:
\[
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\mathcal L(f_\theta(x_i),y_i)
\]
Backpropagation computes \(\nabla_\theta J\) efficiently via chain rule on computation graphs.

For multiclass softmax CE:
\[
p=\text{softmax}(z), \quad \mathcal L=-\sum_c y_c\log p_c
\]
Key result:
\[
\frac{\partial \mathcal L}{\partial z}=p-y
\]
Then for affine layer \(z=Wh+b\):
\[
\frac{\partial\mathcal L}{\partial W}=(p-y)h^\top,
\quad
\frac{\partial\mathcal L}{\partial h}=W^\top(p-y)
\]

Optimization update examples:
- SGD: \(\theta_{t+1}=\theta_t-\eta g_t\)
- AdamW: adaptive moments with decoupled weight decay.

### Why needed for LLMs
LLM pretraining (next-token prediction), supervised fine-tuning, reward optimization, and even many agent-policy updates depend on stable gradients. Without solid understanding of optimization, issues like gradient explosion, divergence at warmup, dead activations, or catastrophic forgetting become impossible to debug.

In large-scale systems, training stability tricks (gradient clipping, weight decay, schedulers, mixed precision scaling) are not “nice-to-have”; they are essential. Week 1 builds the mental model needed to reason about these systems.

### Real-world analogy
Imagine navigating mountain terrain in fog. Loss is altitude above valley floor. Gradient gives steepest descent direction locally. Learning rate is step size. Too large, and you jump across the valley; too small, and you barely move. Momentum is like inertia from previous steps. Adam-like methods adapt steps based on terrain roughness.

### Common mistakes
1. Incorrect target dtype/shape with `nn.CrossEntropyLoss`.
2. Not calling `optimizer.zero_grad()` each step.
3. Confusing training loss reduction with improved generalization.
4. Not monitoring gradient norms, especially in sequence models.
5. Overly aggressive learning rates causing NaNs.
6. Ignoring reproducibility (seed, deterministic config).

### Interview notes
- Derive the \(p-y\) gradient quickly and correctly.
- Explain why AdamW usually outperforms plain SGD in NLP.
- Discuss tradeoffs among batch size, gradient noise, and hardware throughput.
- Mention scheduler warmup importance for Transformer-style training.

---

## Concept 3 — Sequence Modeling, Evaluation, and LLM Readiness

### Layman explanation
Language has order. “The bank approved the loan” and “The loan approved the bank” contain similar words but completely different meaning. Sequence models preserve order and context when generating representations or predicting next tokens. Evaluation then checks whether the model is useful on unseen data—not just memorizing training text.

A practical perspective: model development is not finished when loss goes down. You need metrics, error slicing, and reliability checks before deployment. In agentic AI systems, poor evaluation leads to unsafe tool calls or wrong planning steps.

### Technical explanation
Classical sequence models compute recurrent states:
\[
h_t = f(x_t,h_{t-1})
\]
GRU/LSTM use gates to improve long-range gradient flow. Language modeling objective:
\[
\max_\theta\prod_{t=1}^{T} p(x_t\mid x_{<t};\theta)
\]
Equivalent minimizing token-level NLL:
\[
\mathcal L = -\frac1T\sum_t\log p(x_t\mid x_{<t})
\]
Perplexity:
\[
\text{PPL}=\exp(\mathcal L)
\]

For classification tasks, use accuracy plus macro-F1 (especially with imbalance). For deployment readiness, include confusion matrix, calibration analysis, and slice metrics (length, domain, rare-intent buckets).

### Why needed for LLMs
Transformers replaced recurrence, but the underlying learning goal remains sequence-conditioned prediction. Understanding sequence fundamentals helps with:
- shifted labels in causal LM training,
- context-window truncation effects,
- exposure bias in autoregressive decoding,
- interpretation of perplexity versus downstream task performance.

Evaluation discipline from Week 1 maps directly to LLM and agentic systems: retrieval hit rate, hallucination audits, tool-call precision, and long-horizon success metrics are extensions of the same measurement mindset.

### Real-world analogy
Reading a legal contract: meaning depends on previous clauses. You cannot interpret clause 12 without context from clauses 1–11. Sequence models similarly build context progressively and update internal beliefs token by token.

### Common mistakes
1. Relying only on accuracy for imbalanced datasets.
2. Tuning hyperparameters on test set (data leakage).
3. Reporting perplexity without tokenizer/context details.
4. Ignoring out-of-domain robustness.
5. Treating confidence score as calibrated probability.
6. Not performing qualitative error analysis.

### Interview notes
- Explain difference between intrinsic metrics (NLL/PPL) and task metrics (F1/accuracy).
- Justify macro-F1 for rare intent protection.
- Describe why shifted targets are required for next-token prediction.
- Connect evaluation choices to product risk (e.g., support misrouting, unsafe automation).
