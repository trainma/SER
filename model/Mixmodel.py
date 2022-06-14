"""
Author: Xia hanzhong
Date: 2022-06-05 13:30:38
LastEditors: a1034 a1034084632@outlook.com
LastEditTime: 2022-06-14 20:27:23
FilePath: /Speech-Emotion-Recognition/model/Mixmodel.py
Description: 

"""


import numpy as np
import torch
from torch import nn

from model.Residual import Residual
from model.transformer import TransAm_audio
from model.Residual import ResidualBlock_1DCNN


class MixModel(nn.Module):
    def __init__(
        self,
        d_model,
        batch_size,
        gru_num_layers,
        gru_hidden_size,
        enc_num_layers,
        dropout,
        num_class,
        bidirectional=True,
    ):
        super(MixModel, self).__init__()
        # self.temporal = TemporalBlock(n_inputs=1, n_outputs=1, kernel_size=3, stride=1, dilation=2, padding=1, dropout=dropout)
        self.Residual = Residual(
            hidden_size=2 * gru_hidden_size,
            gru_num_layers=gru_num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.transformer = TransAm_audio(
            feature_size=d_model,
            batch_size=batch_size,
            feature_dim=1576,
            num_layers=enc_num_layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(gru_hidden_size * 2 + d_model, 500)
        self.linear2 = nn.Linear(500, num_class)
        self.leakyRelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.Softmax = nn.Softmax(dim=1)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.bigru = nn.GRU(
            input_size=1576,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, X):
        # X2 = X.clone()
        # X = X.unsqueeze(1)
        # out1 = self.Residual(X)
        # out2 = self.transformer(X2)
        # out = torch.cat((out1.squeeze(0), out2), dim=-1)
        # out = self.dropout(self.leakyRelu(self.linear(out)))
        # out = self.linear2(out)
        # prob = self.Softmax(out)
        # prob = self.logSoftmax(out)

        X = X.unsqueeze(1)
        X = self.Residual(X)
        X2 = X.clone()
        X = X.permute(1, 0, 2)
        out1, _ = self.bigru(X)
        X2 = X2.squeeze(1)
        out2 = self.transformer(X2)
        Out = torch.cat((out1.squeeze(0), out2), dim=-1)

        Out = self.dropout(self.leakyRelu(self.linear(Out)))
        Out = self.linear2(Out)
        return self.Softmax(Out)


if __name__ == "__main__":
    test = torch.randn(64, 1582)
    Model = MixModel(
        d_model=2048,
        batch_size=64,
        gru_num_layers=3,
        gru_hidden_size=128,
        enc_num_layers=1,
        dropout=0.3,
        num_class=7,
    )
    out = Model(test)
    print(out)
