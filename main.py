"""
Author: Xia hanzhong
Date: 2022-06-04 20:42:50
LastEditors: a1034 a1034084632@outlook.com
LastEditTime: 2022-06-14 20:49:20
FilePath: /Speech-Emotion-Recognition/main.py
Description: main.py to train and test the model

"""


import argparse
import time

import torch
import torch.optim as optim
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import extract_feats.librosa as lf
import extract_feats.opensmile as of
from model.Mixmodel import MixModel
from model.transformer import TransAm_audio
from utils import parse_opt, tools

parser = argparse.ArgumentParser(description="PyTorch Time series forecasting")
parser.add_argument("--d_model", type=int, default=2048)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--dec_layers", type=int, default=1)
parser.add_argument("--position", type=str, default=3)
parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=200, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.5,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument("--seed", type=int, default=54321, help="random seed")
parser.add_argument(
    "--log_interval", type=int, default=2000, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="save/model.pt", help="path to save the final model"
)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--normalize", type=int, default=2)
parser.add_argument(
    "--feature_method", type=str, default="o", help="o:using opensmile l:using librosa"
)
parser.add_argument(
    "--save_path", type=str, default="./res", help="the path to save the model"
)
args = parser.parse_args()

Device = torch.device("cuda")
writer = SummaryWriter("runs/scalar_example")


def CreateDataloader(
    train_x, train_y, val_x, val_y, batch_size=args.batch_size, shuffle=True
):
    train_x_tensor = (
        torch.from_numpy(train_x).type(torch.FloatTensor).to(Device)
    )  # (B, N, F, T)
    train_y_tensor = torch.from_numpy(train_y).to(Device)
    val_x_tensor = (
        torch.from_numpy(val_x).type(torch.FloatTensor).to(Device)
    )  # (B, N, F, T)
    val_y_tensor = torch.from_numpy(val_y).to(Device)  # (B, N, T)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    return train_dataloader, val_dataloader


def train(features, y, correct, model):
    output = model(features)
    pred = output
    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    acc = correct / args.batch_size
    loss_train = loss_fn(output, y)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return output, loss_train, acc


def model_test(epoch, features, y, correct, model):
    output = model(features)
    pred = output
    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_acc = correct / args.batch_size
    loss_test = loss_fn(output, y).item()
    return output, loss_test, test_acc


if __name__ == "__main__":
    config = parse_opt()

    if args.feature_method == "l":
        x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    elif args.feature_method == "o":
        x_train, x_test, y_train, y_test = of.load_feature(config, train=True)

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    train_dataloader, val_dataloader = CreateDataloader(
        x_train, y_train, x_test, y_test, batch_size=64
    )
    # model = TransAm_audio(feature_size=args.d_model, batch_size=args.batch_size, feature_dim=1582,num_layers=args.num_layers,
    #                       dropout=args.dropout).to(Device)
    model = MixModel(
        d_model=args.d_model,
        batch_size=64,
        gru_num_layers=3,
        gru_hidden_size=256,
        enc_num_layers=3,
        dropout=0.3,
        num_class=7,
    ).to(Device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # loss_fn = torch.nn.NLLLoss()

    early_stopping = tools.EarlyStopping(patience=7, verbose=True)
    correct = 0
    for epoch in tqdm(range(args.epochs), "training:"):

        start = time.time()
        model.train()
        train_loss = []
        Acc = []
        size = len(train_dataloader.dataset)
        for X, y in train_dataloader:
            output, loss, acc = train(X, y, correct, model)
            train_loss.append(loss.data.item())
            Acc.append(acc)
        time_interval = time.time() - start
        writer.add_scalar("train acc", sum(Acc) / len(Acc), epoch)
        writer.add_scalar("train loss", sum(train_loss) / len(train_loss), epoch)
        print(
            "Epoch {}: train avg loss {:.4f} train Acc {:.4f} train use time {:.4f}".format(
                epoch + 1,
                sum(train_loss) / len(train_loss),
                sum(Acc) / len(Acc),
                time_interval,
            )
        )
        model.eval()
        Test_loss = []
        Acc_Test = []
        with torch.no_grad():
            for X, y in val_dataloader:
                output, test_loss, test_acc = model_test(epoch, X, y, correct, model)
                Test_loss.append(test_loss)
                Acc_Test.append(test_acc)
        writer.add_scalar("val acc", sum(Acc_Test) / len(Acc_Test), epoch)
        writer.add_scalar("val loss", sum(Test_loss) / len(Test_loss), epoch)
        print(
            "Epoch {}: val avg loss {:.4f} val Acc {:.4f}".format(
                epoch + 1,
                sum(Test_loss) / len(Test_loss),
                sum(Acc_Test) / len(Acc_Test),
            )
        )
        early_stopping(sum(Acc_Test) / len(Acc_Test), model, path=args.save_path)
