"""
2019-5-17
Dependencies:
torch: 0.1.11
"""
import torch
import torch.utils.data as Data    #数据进行批训练处理

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5   #一批数据的个数
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)    #把x，y放入数据库
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training   #每次epoch数据的顺序会被打乱
    num_workers=2,              # subprocesses for loading data
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times    #把这批数据整体训练3次
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step     #最后如果数据不够，就把剩下的拿出来
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()