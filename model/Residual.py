import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ResidualBlock_1DCNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ResidualBlock_1DCNN, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # out = self.net(x)
        skip = x.clone()
        x = self.conv1(x)  # 1580
        x = self.chomp1(x)  # 1579
        x = self.relu1(x)
        x = self.dropout1(x)
        # 1579
        x = self.conv2(x)
        x = self.chomp2(x)
        x = self.relu2(x)
        out = self.dropout2(x)

        res = skip if self.downsample is None else self.downsample(x)
        res = res[:, :, :out.shape[-1]]
        return self.relu(out + res)


class Residual(nn.Module):
    def __init__(self, hidden_size, gru_num_layers, dropout, bidirectional=True):
        super(Residual, self).__init__()
        self.dilationConv = weight_norm(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, stride=1))
        self.temporal = ResidualBlock_1DCNN(n_inputs=1, n_outputs=1, kernel_size=3, stride=1, dilation=2, padding=1, dropout=dropout)

        self.dilationConv2 = weight_norm(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=4, stride=2))
        self.BN1 = nn.BatchNorm1d(1)
        self.leakyRelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        # self.bigru = nn.GRU(input_size=785, hidden_size=hidden_size, num_layers=gru_num_layers, dropout=dropout,
        #                     bidirectional=bidirectional)

    def forward(self, X):
        # skip = X.clone()
        # X = self.dilationConv(X)
        # X = self.dropout(self.leakyRelu(X))
        #
        # X = self.dilationConv2(X)
        # X = self.dropout(self.leakyRelu(X))
        X = self.temporal(X)
        X = self.BN1(X)
        X = self.leakyRelu(X)
        X = self.dropout(X)
        # X = X.permute(1, 0, 2)

        # out, _ = self.bigru(X)

        return X


if __name__ == '__main__':
    test_tensor = torch.randn(32, 1, 1582)  # bz channel feature_size
    model = Residual(hidden_size=2 * 128, gru_num_layers=3, dropout=0.2, bidirectional=True)
    # model2 = ResidualBlock_1DCNN(n_inputs=1, n_outputs=1, kernel_size=3, stride=1, dilation=2, padding=1, dropout=0.2)
    out = model(test_tensor)
