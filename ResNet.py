import torch
from torch import nn 
from torch.nn import functional as F

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):
        return X.view(X.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True):
        super(Residual, self).__init__()
        self.stride = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if not same_shape:  # 通过1x1卷积核，步长为2.统一两个特征图shape，保证加法运算正常
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        if self.conv3 is not None:
            X = self.conv3(X)
        return F.relu(X + Y)


class ResNet(nn.Module):
    def __init__(self, classes=1000):
        super(ResNet, self).__init__()
        self.classes = classes
        self.net = nn.Sequential()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.net.add_module("block1", self.b1)
        self.net.add_module("resnet_block_1", self.resnet_block(64, 64, 2, is_first_block=True))
        self.net.add_module("resnet_block_2", self.resnet_block(64, 128, 2))
        self.net.add_module("resnet_block_3", self.resnet_block(128, 256, 2))
        self.net.add_module("resnet_block_4", self.resnet_block(256, 512, 2))
        self.net.add_module("global_avg_pool", GlobalAvgPool2d())
        self.net.add_module("flatten", FlattenLayer())
        self.net.add_module('fc', nn.Linear(512, self.classes))

    def resnet_block(self, in_channels, out_channels, num_residuals, is_first_block=False):
        if is_first_block:
            assert in_channels == out_channels  # 整个模型的第一块的输入通道数等于输出通道数

        block = []
        for i in range(num_residuals):
            if i == 0 and not is_first_block:
                block.append(Residual(in_channels, out_channels, same_shape=False))
            else:
                block.append(Residual(out_channels, out_channels))  # 第一块输入通道数与输出通道数相等
        return nn.Sequential(*block)

    def forward(self, X):
        return self.net(X)
