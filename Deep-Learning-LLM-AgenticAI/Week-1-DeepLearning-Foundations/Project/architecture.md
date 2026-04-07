# Architecture

1. Text input \(x\) is tokenized into ids \(t_{1:T}\).
2. Embedding lookup produces \(E\in\mathbb{R}^{T\times d}\).
3. BiGRU encodes sequence contextual states \(H\in\mathbb{R}^{T\times 2h}\).
4. Masked mean pooling gives sentence vector \(s\in\mathbb{R}^{2h}\).
5. Linear head outputs logits \(z\in\mathbb{R}^{C}\).
6. Softmax + cross-entropy computes training objective.
