# Mathematical Explanation — Week 1 Project

## Notation
- \(B\): batch size
- \(T\): sequence length
- \(V\): vocabulary size
- \(d\): embedding dimension
- \(h\): GRU hidden dimension
- \(C\): number of intent classes

## Forward pass
1. **Token embedding**
\[
X_b = [E_{t_1},\dots,E_{t_T}]\in\mathbb R^{T\times d}
\]
2. **BiGRU encoding**
\[
H_b = \text{BiGRU}(X_b)\in\mathbb R^{T\times 2h}
\]
3. **Masked mean pooling**
\[
s_b = \frac{\sum_{i=1}^{T} m_{b,i} h_{b,i}}{\sum_{i=1}^{T}m_{b,i}}\in\mathbb R^{2h}
\]
4. **Classifier logits**
\[
z_b = W s_b + b\in\mathbb R^{C}
\]
5. **Probabilities**
\[
p_b = \text{softmax}(z_b)
\]

## Loss
\[
\mathcal J = -\frac1B\sum_{b=1}^{B}\log p_{b,y_b}
\]

## Backprop key equations
For sample \(b\):
\[
\frac{\partial\mathcal L_b}{\partial z_b}=p_b-y_b^{(onehot)}
\]
\[
\frac{\partial\mathcal J}{\partial W}=\frac1B\sum_b (p_b-y_b)s_b^\top
\]
\[
\frac{\partial\mathcal J}{\partial s_b}=W^\top(p_b-y_b)
\]
Pooled representation gradient distributes back through mask-normalized average to each time step, then through GRU recurrence and embedding rows.

## Optimization
Use AdamW update per parameter tensor \(\theta\):
\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]
with decoupled weight decay.

## Practical interpretation
- If class probability for true label is low, gradient magnitude is high.
- Rare intents often need macro-F1 tracking to avoid collapse into frequent classes.
