'''import torch
import torch.nn.functional as F   #nn是他的神经网络模块
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)   #torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。
#torch只会处理二位数据
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  #最主要的功能；搭建网络层所需要的信息
        super(Net, self).__init__()   #继承Net到那个模块；固定操作
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  #输入，输出大小
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输入，输出大小



    def forward(self,x):   #最主要的功能；网络前向传播的过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1)
print(net)   #把网络的层结构全部告诉你

plt.ion()   #让plt实时打印

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  #把网络的参数属性放入更新
loss_func = torch.nn.MSELoss()  #均方误差计算损失函数

for t in range(100):
    prediction = net(x)   #放输入

    loss = loss_func(prediction, y)  #真实值在后面

    optimizer.zero_grad()   #把所有参数的梯度设为0
    loss.backward()   #反向传播
    optimizer.step()   #对参数进行优化
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()   #?
        plt.scatter(x.data.numpy(), y.data.numpy())   #
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)   #红色 线宽5
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})   #坐标 文字
        plt.pause(0.1)

plt.ioff()   #
plt.show()'''

"""
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)   #拓展维度
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting   #交互模式

for t in range(200):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    print(y.dim())

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
#plt.show()



