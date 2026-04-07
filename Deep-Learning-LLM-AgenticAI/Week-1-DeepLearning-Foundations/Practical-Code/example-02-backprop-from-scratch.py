"""Problem: Verify autograd gradients against manual gradients on binary sentiment task formulation."""
import torch

torch.manual_seed(42)
X = torch.randn(8, 5)
y = torch.randint(0, 2, (8, 1)).float()
W = torch.randn(5, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Forward
logits = X @ W + b
probs = torch.sigmoid(logits)
loss = -(y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)).mean()
loss.backward()

# Manual gradient derivation: dL/dz = (p-y)/N
manual_dz = (probs.detach() - y) / y.size(0)
manual_dW = X.t() @ manual_dz
manual_db = manual_dz.sum()

print("Autograd dW close:", torch.allclose(W.grad, manual_dW, atol=1e-5))
print("Autograd db close:", torch.allclose(b.grad, manual_db, atol=1e-5))
