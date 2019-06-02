'''import torch
from torch.autograd import Variable   #使用变量要用这个包


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor,requires_grad=True)    #requires_grad=True通过variables搭建计算图纸，使节点计算反向传播的梯度

print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()   #让v_out反向传播
print(variable.grad)   #看variable的梯度属性
print(variable.data)  #看variable的data属性

print(variable.data.numpy())'''

"""
Dependencies:
torch: 0.1.11
"""
import torch
from torch.autograd import Variable   #使用变量要用这个包

# Variable in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in Tensorflow or Theano.
# So torch does not have placeholder, torch can just pass variable to the computational graph.

tensor = torch.FloatTensor([[1,2],[3,4]])            # build a tensor
variable = Variable(tensor, requires_grad=True)      # build a variable, usually for compute gradients

print(tensor)       # [torch.FloatTensor of size 2x2]
print(variable)     # [torch.FloatTensor of size 2x2]

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5

v_out.backward()    # backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)
'''
 0.5000  1.0000
 1.5000  2.0000
'''

print(variable)     # this is data in variable format
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # this is data in tensor format
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy format
"""
[[ 1.  2.]
 [ 3.  4.]]
"""