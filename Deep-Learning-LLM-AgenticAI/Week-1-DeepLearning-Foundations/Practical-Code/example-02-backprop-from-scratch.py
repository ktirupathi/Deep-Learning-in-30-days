"""
Example 02: Manual vs autograd gradients for BCE logistic regression.
Dataset explanation:
- Uses synthetic batch to isolate gradient mechanics exactly.
- This is the mathematical microscope behind NLP classifiers.

Architecture:
- Single linear logit z = XW + b with sigmoid + BCE.
"""
import torch

torch.manual_seed(7)
X = torch.randn(10, 6)
y = torch.randint(0, 2, (10, 1)).float()
W = torch.randn(6, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

z = X @ W + b
p = torch.sigmoid(z)
loss = -(y * torch.log(p + 1e-8) + (1 - y) * torch.log(1 - p + 1e-8)).mean()
loss.backward()

# Math-to-code mapping: dL/dz = (p - y)/B
B = y.size(0)
dz = (p.detach() - y) / B
manual_dW = X.t() @ dz
manual_db = dz.sum()

print("Tensor shapes: X", X.shape, "W", W.shape, "z", z.shape)
print("autograd dW == manual dW:", torch.allclose(W.grad, manual_dW, atol=1e-6))
print("autograd db == manual db:", torch.allclose(b.grad, manual_db, atol=1e-6))
print("Expected output: both checks True")
print("Improvements: finite-difference gradient check, vectorized Hessian approximation.")
