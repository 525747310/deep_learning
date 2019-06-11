import torch
from torch.autograd import Variable

grad_list = []


def print_grad(grad):
    grad_list.append(grad)


x = Variable(torch.randn(2, 1), requires_grad=True)
y = x + 2
z = torch.mean(torch.pow(y, 2))
lr = 1e-3
y.register_hook(print_grad)
z.backward()
x.data -= lr * x.grad.data

print(grad_list)