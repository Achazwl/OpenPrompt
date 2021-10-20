
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import  AdamW, get_linear_schedule_with_warmup

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 5)
        self.b = nn.Linear(5, 3)

    def forward(self, x):
        return self.b(x)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = 1e-3)


class A(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x):
        return self.a(x)

batch = 1
test = 1 # 0
b = B()
a = A(b.a)
if test == 0:
    x = torch.rand((batch, 5))
    y = torch.rand((batch, 3))
    loss = (b(x)-y)**2
    loss.backward(loss)
    assert(b.a.weight.grad is None)
    assert(b.b.weight.grad is not None)
else:
    x = torch.rand((batch, 3))
    y = torch.rand((batch, 3))
    optimizer = b.configure_optimizers()
    optimizer.zero_grad()
    loss = (b(a(x))-y)**2
    loss.backward(loss)
    assert(b.a.weight.grad is not None)
    assert(b.b.weight.grad is not None)
    assert((b.a.weight.grad == a.a.weight.grad).all())
    assert(b.a is a.a)
    from IPython import embed; embed()
    optimizer.step()
    assert((b.a.weight == a.a.weight).all())
    assert(b.a is a.a)
    from IPython import embed; embed()

