# Week 1 Mathematics — Full Derivations Without Skipped Steps

## Topic 1: Softmax + Cross-Entropy for Multiclass Classification

### 1) Scalar form
Let logits \(z_1,\dots,z_C\), target class \(y\).
\[
p_i = \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}},\quad \mathcal L = -\log p_y
\]

### 2) Vector form
\[
\mathbf p = \text{softmax}(\mathbf z),\quad \mathcal L = -\sum_{i=1}^{C} y_i\log p_i
\]
where \(\mathbf y\) is one-hot.

### 3) Matrix form (batch)
For batch size \(B\): logits \(Z\in\mathbb R^{B\times C}\), probabilities \(P\in\mathbb R^{B\times C}\), labels \(Y\in\{0,1\}^{B\times C}\).
\[
\mathcal J = -\frac1B\sum_{b=1}^{B}\sum_{c=1}^{C}Y_{bc}\log P_{bc}
\]

### 4) Dimension analysis
- Input features: \(X\in\mathbb R^{B\times d}\)
- Weights: \(W\in\mathbb R^{d\times C}\)
- Bias: \(b\in\mathbb R^{C}\)
- Logits: \(Z=XW+\mathbf 1b^\top\in\mathbb R^{B\times C}\)

### 5) Forward pass math
\[
Z = XW+b,\quad P = \text{softmax}(Z)
\]

### 6) Loss function math
\[
\mathcal J(W,b)= -\frac1B\sum_{b,c}Y_{bc}\log P_{bc}
\]

### 7) Backpropagation derivation
For one sample, \(\mathcal L=-\sum_i y_i\log p_i\), \(p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}\).
\[
\frac{\partial\mathcal L}{\partial z_k}=\sum_i\frac{\partial\mathcal L}{\partial p_i}\frac{\partial p_i}{\partial z_k}
\]
\[
\frac{\partial\mathcal L}{\partial p_i}=-\frac{y_i}{p_i},\quad
\frac{\partial p_i}{\partial z_k}=p_i(\delta_{ik}-p_k)
\]
Substitute:
\[
\frac{\partial\mathcal L}{\partial z_k}=
\sum_i -\frac{y_i}{p_i}p_i(\delta_{ik}-p_k)
= -y_k + p_k\sum_i y_i = p_k-y_k
\]

### 8) Chain rule expansion to weights
\[
Z=XW+b\Rightarrow
\frac{\partial\mathcal J}{\partial W}=\frac1B X^\top(P-Y),\quad
\frac{\partial\mathcal J}{\partial b}=\frac1B\sum_{b=1}^{B}(P-Y)_b
\]

### 9) Final gradient formula
\[
\nabla_W\mathcal J = \frac{1}{B}X^\top(P-Y)
\]

### 10) Numerical example
Take \(z=[2,1,0.1]\), true class is first class.
\[
e^z=[7.389,2.718,1.105],\; S=11.212,\; p=[0.659,0.242,0.099]
\]
\[
\mathcal L=-\log(0.659)=0.417
\]
Gradient wrt logits:
\[
\nabla_z\mathcal L=[-0.341,0.242,0.099]
\]

### 11) Variables explained
- \(C\): number of classes
- \(B\): batch size
- \(d\): feature dimension
- \(X\): input matrix
- \(W,b\): trainable parameters
- \(Z\): logits
- \(P\): predicted probabilities
- \(Y\): one-hot labels

---

## Topic 2: Binary Cross-Entropy with Sigmoid

### 1) Scalar form
\[
\hat y=\sigma(z)=\frac1{1+e^{-z}},\quad
\mathcal L=-(y\log\hat y+(1-y)\log(1-\hat y))
\]

### 2) Vector form
For batch vectors \(\hat{\mathbf y},\mathbf y\in\mathbb R^{B}\):
\[
\mathcal J=-\frac1B\sum_{i=1}^{B}\big[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)\big]
\]

### 3) Matrix form
With features \(X\in\mathbb R^{B\times d}\), \(w\in\mathbb R^{d}\):
\[
\mathbf z=Xw+b\mathbf 1,\; \hat{\mathbf y}=\sigma(\mathbf z)
\]

### 4) Dimension analysis
- \(X\): \(B\times d\)
- \(w\): \(d\times 1\)
- \(z,\hat y\): \(B\times 1\)

### 5) Forward pass math
\[
z_i = x_i^\top w + b,\quad \hat y_i = \sigma(z_i)
\]

### 6) Loss function math
\[
\mathcal J(w,b)=\frac1B\sum_i\mathcal L_i
\]

### 7) Backpropagation derivation
For sample \(i\):
\[
\frac{\partial \mathcal L_i}{\partial z_i}=\hat y_i-y_i
\]

### 8) Chain rule expansion
\[
\frac{\partial z_i}{\partial w}=x_i,\quad
\frac{\partial z_i}{\partial b}=1
\]
\[
\nabla_w\mathcal J = \frac1B\sum_i (\hat y_i-y_i)x_i,
\quad
\frac{\partial\mathcal J}{\partial b}=\frac1B\sum_i(\hat y_i-y_i)
\]

### 9) Final gradient formula
\[
\nabla_w\mathcal J = \frac1B X^\top(\hat{\mathbf y}-\mathbf y)
\]

### 10) Numerical example
If \(y=1,z=0.4\): \(\hat y=0.5987\), loss \(=-\log(0.5987)=0.513\), gradient \(\hat y-y=-0.4013\).

### 11) Variables explained
- \(w,b\): logistic regression parameters
- \(x_i\): feature vector
- \(z_i\): logit
- \(\hat y_i\): predicted probability

---

## Topic 3: Embedding + Mean Pooling + Linear Head

### 1) Scalar form
Token id \(t\) selects row \(E_t\). For sequence length \(T\):
\[
s = \frac{1}{T}\sum_{i=1}^{T}E_{t_i}
\]
\[
z = Ws+b
\]

### 2) Vector form
\(s\in\mathbb R^d\), \(z\in\mathbb R^C\).

### 3) Matrix form
Embedding lookup returns \(H\in\mathbb R^{T\times d}\), pooling uses mask \(m\in\{0,1\}^{T}\):
\[
s=\frac{\sum_i m_i h_i}{\sum_i m_i}
\]

### 4) Dimension analysis
- \(H\): \(T\times d\)
- \(s\): \(d\)
- \(W\): \(C\times d\)
- \(z\): \(C\)

### 5) Forward pass math
\[
z = Ws+b,\quad p=\text{softmax}(z)
\]

### 6) Loss function math
\[
\mathcal L = -\log p_y
\]

### 7) Backpropagation derivation
\[
\frac{\partial\mathcal L}{\partial z}=p-y
\]
\[
\frac{\partial\mathcal L}{\partial s}=W^\top(p-y)
\]

### 8) Chain rule expansion to embeddings
\[
\frac{\partial s}{\partial h_i}=\frac{m_i}{\sum_k m_k}I_d
\]
\[
\frac{\partial \mathcal L}{\partial h_i}=
\frac{m_i}{\sum_k m_k}\frac{\partial\mathcal L}{\partial s}
\]

### 9) Final gradient formula for row \(E_j\)
\[
\frac{\partial \mathcal L}{\partial E_j}=\sum_{i:t_i=j}
\frac{m_i}{\sum_k m_k}\frac{\partial \mathcal L}{\partial s}
\]

### 10) Numerical example
Tokens [“great”, “movie”], d=2:
\(E_{great}=[1,2], E_{movie}=[3,1]\), pooled \(s=[2,1.5]\).
With
\(W=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}\),
\(z=[2.75,-1.25]\).

### 11) Variables explained
- \(E\): embedding matrix
- \(m\): attention/padding mask
- \(h_i\): token embedding at position \(i\)
- \(s\): pooled sentence embedding

---

## Topic 4: Gradient Descent and Adam Update Equations

### 1) Scalar form (SGD)
\[
\theta_{t+1}=\theta_t-\eta g_t
\]

### 2) Vector form
Same equation element-wise for vector parameters.

### 3) Matrix form
For tensor parameters, update applies element-wise over entire tensor.

### 4) Dimension analysis
- \(\theta\): any parameter tensor shape
- \(g_t\): same shape as \(\theta\)

### 5) Forward pass math
Not model-specific; optimizer acts after gradient computation.

### 6) Loss function math
Any differentiable objective \(\mathcal J(\theta)\).

### 7) Backpropagation derivation
Backprop provides \(g_t=\nabla_\theta \mathcal J\).

### 8) Chain rule expansion
Handled by autograd graph from output loss to parameters.

### 9) Final gradient/update formula (Adam)
\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]
\[
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]
\[
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

### 10) Numerical example
For one parameter: \(g_t=0.2,\beta_1=0.9,\beta_2=0.999,\eta=10^{-3}\), zero-initialized moments:
\(m_t=0.02, v_t=0.00004, \hat m_t=0.2, \hat v_t=0.04\), update \(\Delta\theta\approx-0.001\).

### 11) Variables explained
- \(\eta\): learning rate
- \(\beta_1,\beta_2\): decay rates
- \(m_t,v_t\): first/second moments

---

## Topic 5: Language Modeling NLL and Perplexity

### 1) Scalar form
\[
\mathcal L_t=-\log p(x_t|x_{<t})
\]

### 2) Vector form
\[
\mathcal L_{seq}=\frac1T\sum_{t=1}^{T}\mathcal L_t
\]

### 3) Matrix form
For logits \(Z\in\mathbb R^{B\times T\times V}\), flatten to \((B\cdot T,V)\) with labels \((B\cdot T)\).

### 4) Dimension analysis
- \(B\): batch size
- \(T\): sequence length
- \(V\): vocabulary size

### 5) Forward pass math
\[
p(x_t|x_{<t})=\text{softmax}(z_t)_{x_t}
\]

### 6) Loss function math
\[
\mathcal J = -\frac1{BT}\sum_{b,t}\log p(x_{b,t}|x_{b,<t})
\]

### 7) Backpropagation derivation
Gradient wrt token logits again yields \(p-y\) at each time step.

### 8) Chain rule expansion
Gradients flow from output projection to recurrent/attention blocks and embeddings.

### 9) Final formula
\[
\text{PPL}=\exp\left(\frac{1}{T}\sum_t -\log p(x_t|x_{<t})\right)
\]

### 10) Numerical example
If token losses are [1.2, 0.9, 1.1], average NLL = 1.067, PPL \(=e^{1.067}=2.91\).

### 11) Variables explained
- \(x_t\): true token at step \(t\)
- \(z_t\): logits at step \(t\)
- \(\mathcal L_t\): token NLL
