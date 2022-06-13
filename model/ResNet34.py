from torch import nn
import torch as t
from torch.nn import functional as F
from torch.autograd import Variable as V


class ResidualBlock(nn.Module):  # 定义ResidualBlock类 （11）
    """实现子modual：residualblock"""

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):  # 初始化，自动执行 （12）
        super(ResidualBlock, self).__init__()  # 继承nn.Module （13）
        self.left = nn.Sequential(  # 左网络，构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （14）（31）
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut  # 右网络，也属于Sequential，见（8）步可知，并且充当残差和非残差的判断标志。 （15）

    def forward(self, x):  # ResidualBlock的前向传播函数 （29）
        out = self.left(x)  # # 和调用forward一样如此调用left这个Sequential（30）
        if self.right is None:  # 残差（ResidualBlock）（32）
            residual = x  # （33）
        else:  # 非残差（非ResidualBlock） （34）
            residual = self.right(x)  # （35）
        out += residual  # 结果相加 （36）
        # print(out.size())  # 检查每单元的输出的通道数 （37）
        return F.relu(out)  # 返回激活函数执行后的结果作为下个单元的输入 （38）


class ResNet(nn.Module):  # 定义ResNet类，也就是构建残差网络结构 （2）
    """实现主module：ResNet34"""

    def __init__(self, numclasses=1000, dropout=.2):  # 创建实例时直接初始化 （3）
        super(ResNet, self).__init__()  # 表示ResNet继承nn.Module （4）
        self.pre = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （5）（26）
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),  # 卷积层，输入通道数为3，输出通道数为64，包含在Sequential的子module，层层按顺序自动执行
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer1 = self.make_layer(64, 128,
                                      4)
        # 输入通道数为64，输出为128，根据残差网络结构将一个非Residual Block加上多个Residual Block构造成一层layer（6）
        self.layer2 = self.make_layer(128, 256, 4, stride=2)  # 输入通道数为128，输出为256 （18，流程重复所以标注省略7-17过程）
        self.layer3 = self.make_layer(256, 256, 6, stride=2)  # 输入通道数为256，输出为256 （19，流程重复所以标注省略7-17过程）
        self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 输入通道数为256，输出为512 （20，流程重复所以标注省略7-17过程）

    #        self.fc = nn.Linear(512, numclasses)  # 全连接层，属于残差网络结构的最后一层，输入通道数为512，输出为numclasses （21）

    def make_layer(self, inchannel, outchannel, block_num,
                   stride=1):  # 创建layer层，（block_num-1）表示此层中Residual Block的个数 （7）
        """构建layer，包含多个residualblock"""
        shortcut = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行 （8）
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []  # 创建一个列表，将非Residual Block和多个Residual Block装进去 （9）
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))  # 非残差也就是非Residual Block创建及入列表 （10）

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))  # 残差也就是Residual Block创建及入列表 （16）

        return nn.Sequential(
            *layers)  # 通过nn.Sequential函数将列表通过非关键字参数的形式传入，并构成一个新的网络结构以Sequential形式构成，一个非Residual Block和多个Residual Block分别成为此Sequential的子module，层层按顺序自动执行，并且类似于forward前向传播函数，同样的方式调用执行 （17） （28）

    def forward(self, x):  # ResNet类的前向传播函数 （24）
        x = self.pre(x)  # 和调用forward一样如此调用pre这个Sequential（25）

        x = self.layer1(x)  # 和调用forward一样如此调用layer1这个Sequential（27）
        x = self.layer2(x)  # 和调用forward一样如此调用layer2这个Sequential（39，流程重复所以标注省略28-38过程）
        x = self.layer3(x)  # 和调用forward一样如此调用layer3这个Sequential（40，流程重复所以标注省略28-38过程）
        x = self.layer4(x)  # 和调用forward一样如此调用layer4这个Sequential（41，流程重复所以标注省略28-38过程）
        x = self.dropout(x)

        x = F.avg_pool2d(x, 3, padding=1)  # 平均池化 （42）
        x = x.view(x.size(0), -1)  # 设置返回结果的尺度 （43）
        # return self.fc(x)  # 返回结果 （44）
        return x


if __name__ == '__main__':
    model = ResNet()  # 创建ResNet残差网络结构的模型的实例  (1)
    input = V(t.randn(1, 1, 50, 616))  # 输入数据的创建，注意要报证通道数与残差网络结构每层需要的通道数一致，此数据通道数为3 （22）
    output = model(input)  # 把数据输入残差模型，等同于开始调用ResNet类的前向传播函数 （23）
    print(output)  # 输出运行的结果 （45）
