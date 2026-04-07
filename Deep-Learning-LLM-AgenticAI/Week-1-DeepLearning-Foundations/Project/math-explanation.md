# Mathematical Explanation

For utterance tokens \(t_{1:T}\):
\[
X = [E_{t_1}, \ldots, E_{t_T}]
\]
BiGRU hidden states:
\[
H = \text{BiGRU}(X)
\]
Masked pooling:
\[
s = \frac{\sum_{i=1}^{T} m_i h_i}{\sum_{i=1}^{T} m_i}
\]
Classifier:
\[
z = Ws+b,\quad p=\text{softmax}(z)
\]
Loss for class \(y\):
\[
\mathcal{L}=-\log p_y
\]
Gradient descent update:
\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]
