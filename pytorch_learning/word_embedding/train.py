# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#教程网址:https://my.oschina.net/earnp/blog/1113896

#首先我们需要word_to_ix = {'hello': 0, 'world': 1}，每个单词我们需要用一个数字去表示他，这样我们需要hello的时候，就用0来表示它。
word_to_ix = {'hello': 0, 'world': 1}
#接着就是word embedding的定义nn.Embedding(2, 5)，这里的2表示有2个词，5表示5维度，其实也就是一个2x5的矩阵，所以如果你有1000个词，每个词希望是100维，你就可以这样建立一个word embedding，nn.Embedding(1000, 100)。如何访问每一个词的词向量是下面两行的代码，注意这里的词向量的建立只是初始的词向量，并没有经过任何修改优化，我们需要建立神经网络通过learning的办法修改word embedding里面的参数使得word embedding每一个词向量能够表示每一个不同的词。
embeds = nn.Embedding(2, 5)
hello_idx = torch.LongTensor([word_to_ix['hello']])
#接着这两行代码表示得到一个Variable，它的值是hello这个词的index，也就是0。这里要特别注意一下我们需要Variable，因为我们需要访问nn.Embedding里面定义的元素，并且word embeding算是神经网络里面的参数，所以我们需要定义Variable
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
#hello_embed = embeds(hello_idx)这一行表示得到word embedding里面关于hello这个词的初始词向量，最后我们就可以print出来。
print('hello_embed是：',hello_embed)

world_idx = torch.LongTensor([word_to_ix['world']])
world_idx = Variable(world_idx)
world_embed = embeds(world_idx)
print('world_embed是：',world_embed)

