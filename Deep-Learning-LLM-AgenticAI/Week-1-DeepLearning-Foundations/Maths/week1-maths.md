# Week 1 Maths — Derivations for NLP Deep Learning

## 1) Affine layer and dimensions
Given batch \(X\in\mathbb{R}^{B\times d_{in}}\), weights \(W\in\mathbb{R}^{d_{in}\times d_{out}}\), bias \(b\in\mathbb{R}^{d_{out}}\):
\[
Z = XW + \mathbf{1}b^\top
\]
where \(Z\in\mathbb{R}^{B\times d_{out}}\).

## 2) Softmax + Cross-Entropy (multiclass)
For logits \(z\in\mathbb{R}^K\):
\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]
For one-hot label \(y\):
\[
\mathcal{L} = -\sum_i y_i\log p_i = -\log p_{y^*}
\]
Gradient wrt logits:
\[
\frac{\partial \mathcal{L}}{\partial z_i}=p_i-y_i
\]
This identity is crucial in efficient implementations.

## 3) Chain rule through 2-layer MLP
\[
h = \phi(XW_1+b_1),\quad \hat{Y}=hW_2+b_2
\]
\[
\frac{\partial \mathcal{L}}{\partial W_2}=h^\top\frac{\partial\mathcal{L}}{\partial\hat{Y}},\quad
\frac{\partial \mathcal{L}}{\partial h}=\frac{\partial\mathcal{L}}{\partial\hat{Y}}W_2^\top
\]
For ReLU: \(\phi'(u)=\mathbb{1}[u>0]\).

## 4) Embedding gradients
Embedding matrix \(E\in\mathbb{R}^{V\times d}\). Token index \(t\) selects row \(E_t\).
Only selected rows receive gradients:
\[
\frac{\partial \mathcal{L}}{\partial E_j}=0 \text{ if token } j \text{ absent in batch}
\]
This sparse update explains embedding efficiency.

## 5) Binary cross-entropy
For binary label \(y\in\{0,1\}\), logit \(z\), \(\sigma(z)=1/(1+e^{-z})\):
\[
\mathcal{L}=-(y\log\sigma(z)+(1-y)\log(1-\sigma(z)))
\]
\[
\frac{\partial\mathcal{L}}{\partial z}=\sigma(z)-y
\]

## 6) Gradient descent and Adam
SGD step:
\[
\theta_{t+1}=\theta_t-\eta g_t
\]
Adam:
\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,
\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
\]
\[
\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t)
\]
\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

## 7) Perplexity for language models
For tokens \(x_{1:T}\):
\[
\mathcal{L}_{NLL} = -\frac{1}{T}\sum_{t=1}^{T}\log p(x_t|x_{<t})
\]
Perplexity:
\[
\text{PPL}=\exp(\mathcal{L}_{NLL})
\]
Lower is better.

## 8) Attention preview math (for Week 2 bridge)
Single-head attention:
\[
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]
Scale \(1/\sqrt{d_k}\) stabilizes gradients/logits magnitude.

## 9) Numerical example (softmax + CE)
Let logits \([2.0, 1.0, 0.1]\), true class index 0.
\[
e^z=[7.389,2.718,1.105],\quad \sum=11.212
\]
\[
p=[0.659,0.242,0.099]
\]
Loss:
\[
\mathcal{L}=-\log(0.659)=0.417
\]
Logit gradients:
\[
\nabla_z \mathcal{L}= [0.659-1, 0.242-0, 0.099-0]=[-0.341,0.242,0.099]
\]

## 10) Matrix checklist for implementation
- Inputs: `LongTensor [B, T]` for token ids
- Embeddings: `FloatTensor [B, T, d]`
- Pooled sentence vector: `[B, d]`
- Classifier logits: `[B, C]`
- CE labels: `[B]` (class indices)
