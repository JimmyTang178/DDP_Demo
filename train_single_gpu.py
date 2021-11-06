"""
Adapted from https://github.com/wmpscc/CNN-Series-Getting-Started-and-PyTorch-Implementation
"""
import torch
import torchvision.transforms as transforms
import argparse
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
from tqdm import tqdm

from ResNet import ResNet
from utils import evaluate_accuracy


def main(opt):
    """
    Train and valid
    """
    batch_size = opt.batch_size
    device = torch.device('cuda', opt.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    #加载CIFAR10数据
    transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
    #定义模型
    resnet = ResNet(classes=opt.num_classes)
    resnet = resnet.to(device)
    #损失函数
    optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)
    lossFN = nn.CrossEntropyLoss()

    num_epochs = opt.epoch
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = resnet(X)

            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        if epoch > 0 and epoch % 2 == 0:
            test_acc = evaluate_accuracy(val_loader, resnet)
            print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Single GPU training script.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='select GPU devices')
    opt = parser.parse_args()
    print("opt:", opt)
    main(opt)

    
