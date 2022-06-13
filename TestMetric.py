import torch
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from torch.utils.data import TensorDataset, DataLoader
from model.Mixmodel import MixModel

Device = torch.device("cpu")
import extract_feats.opensmile as of
from utils import parse_opt, tools


def CreateDataloader(train_x, train_y, val_x, val_y, batch_size, shuffle=True):
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(Device)  # (B, N, F, T)
    train_y_tensor = torch.from_numpy(train_y).to(Device)
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(Device)  # (B, N, F, T)
    val_y_tensor = torch.from_numpy(val_y).to(Device)  # (B, N, T)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    model = MixModel(d_model=2048, batch_size=64, gru_num_layers=3, gru_hidden_size=256, enc_num_layers=3, dropout=0.4,
                     num_class=7).to(Device)
    model.load_state_dict(torch.load('./res/checkpoint.pth'))
    config = parse_opt()
    x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    train_dataloader, val_dataloader = CreateDataloader(x_train, y_train, x_test, y_test, batch_size=64)
    correct = 0
    # [ "angry", "calm", "disgust", "fearful", "happy","neutral", "sad","surprised" ]
    angryList = [0, 0, 0, 0, 0, 0, 0, 0]
    calmList = [0, 0, 0, 0, 0, 0, 0, 0]
    disgustList = [0, 0, 0, 0, 0, 0, 0, 0]
    fearfulList = [0, 0, 0, 0, 0, 0, 0, 0]
    happyList = [0, 0, 0, 0, 0, 0, 0, 0]
    neutralList = [0, 0, 0, 0, 0, 0, 0, 0]
    sadList = [0, 0, 0, 0, 0, 0, 0, 0]
    suprisedList = [0, 0, 0, 0, 0, 0, 0, 0]

    for batch, (X, y) in enumerate(val_dataloader):
        output = model(X)
        pred = output
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        predLst = list(pred.argmax(1).cpu().numpy())
        CorrectLst = list(y.argmax(1).cpu().numpy())

        for i in range(len(predLst)):
            if CorrectLst[i] == 0:
                angryList[predLst[i]] += 1
            if CorrectLst[i] == 1:
                calmList[predLst[i]] += 1
            if CorrectLst[i] == 2:
                disgustList[predLst[i]] += 1
            if CorrectLst[i] == 3:
                fearfulList[predLst[i]] += 1
            if CorrectLst[i] == 4:
                happyList[predLst[i]] += 1
            if CorrectLst[i] == 5:
                neutralList[predLst[i]] += 1
            if CorrectLst[i] == 6:
                sadList[predLst[i]] += 1
            if CorrectLst[i] == 7:
                suprisedList[predLst[i]] += 1
    print("angryList: ", angryList)
    print("calmList: ", calmList)
    print("disgustList: ", disgustList)
    print("fearfulList: ", fearfulList)
    print("happyList: ", happyList)
    print("neutralList: ", neutralList)
    print("sadList: ",sadList)
    print("suprisedList: ",suprisedList)
