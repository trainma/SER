from sklearn.utils import shuffle
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
#### positional encoding ####
'''
transformer中默认的位置编码方式
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0) former
        pe = pe.unsqueeze(0).transpose(0,1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        #return x+Variable(self.pe[:x.size(0),:x.size(1), :], requires_grad=False)

        return x+Variable(self.pe[:x.size(0), :], requires_grad=False)

'''
可学习的位置编码方式
'''
class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.5, 0.5)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + Variable(self.pe[:x.size(0), :], 
                         requires_grad=False)
        return self.dropout(x)
    
'''
TBD:相对位置编码
'''

"""
因果卷积
"""

class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,embedding_size2=16,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = weight_norm(CausalConv1d(in_channels,embedding_size,kernel_size=k))
        # self.causal_convolution2 = weight_norm(CausalConv1d(in_channels,embedding_size,kernel_size=k))
        self.init_weights()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weights(self):
        self.causal_convolution.weight.data.normal_(0, 0.01)
        #self.causal_convolution2.weight.data.normal_(0, 0.01)

    def forward(self,x):
        #original type
        # if x.shape[0]==32:
        #     x = self.causal_convolution(x)
        # else:
        #     x= self.causal_convolution2(x)
        return torch.tanh(self.causal_convolution(x))

        # if x.shape[1]==32:
        #     x_1=F.tanh(self.causal_convolution(x[0:0+150])) #torch.Size([150, 32, 1])
        #     x_2=F.tanh(self.causal_convolution(x[150:150+150]))
        #     x_3=F.tanh(self.causal_convolution(x[300:300+150]))
        #     x_4=F.tanh(self.causal_convolution(x[450:450+150]))
        # else:
        #     x_1=F.tanh(self.causal_convolution2(x[0:0+150])) #torch.Size([150, 32, 1])
        #     x_2=F.tanh(self.causal_convolution2(x[150:150+150]))
        #     x_3=F.tanh(self.causal_convolution2(x[300:300+150]))
        #     x_4=F.tanh(self.causal_convolution2(x[450:450+150]))
        # return torch.cat((x_1,x_2,x_3,x_4),0)

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=2, #3
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))