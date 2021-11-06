import torch
import torchvision.transforms as transforms
import argparse
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets
import torch.distributed as dist  #1. DDP相关包
import torch.utils.data.distributed

from tqdm import tqdm

from ResNet import ResNet
from utils import evaluate_accuracy


def main(opt):
    """
    Train and valid
    """
    
    dist.init_process_group(backend='nccl', init_method=opt.init_method) #4.初始化进程组，采用nccl后端

    batch_size = opt.batch_size
    device = torch.device('cuda', opt.local_rank if torch.cuda.is_available() else 'cpu')
    print("Using device:{}\n".format(device))

    transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)  #5. 分配数据，将数据集划分为N份，每个GPU一份
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=opt.num_workers, sampler=train_sampler) #注意要loader里面要指定sampler，这样才能将数据分发到多个GPU上
    nb = len(train_loader)
    
    #6.一般只在主进程进行验证，所以在local_rank=-1或者0的时候才实例化val_loader
    if opt.local_rank in [-1, 0]:
        valset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
        

    resnet = ResNet(classes=opt.num_classes)
    resnet = resnet.to(device)
    resnet = torch.nn.parallel.DistributedDataParallel(resnet, device_ids=[opt.local_rank], output_device=opt.local_rank) #7. 将模型包装成分布式

    optimizer = optim.Adam(resnet.parameters(), lr=opt.lr)
    cross_entropy = nn.CrossEntropyLoss()
    
    num_epochs = opt.epoch
    for epoch in range(num_epochs):
        if opt.local_rank != -1:
            train_loader.sampler.set_epoch(epoch) #不同的epoch设置不同的随机数种子，打乱数据

        loader = enumerate(train_loader)
        if opt.local_rank in [-1, 0]:
            loader = tqdm(loader, total=nb)  #只在主进程打印进度条

        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for _, (X, y) in loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = resnet(X)

            loss = cross_entropy(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        if opt.local_rank in [-1, 0] and epoch % 2 == 0 and epoch > 0:
            test_acc = evaluate_accuracy(val_loader, resnet)
            print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP training script.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank of current process') #2. 指定local_rank
    parser.add_argument('--init_method', default='env://') #3.指定初始化方式，这里用的是环境变量的初始化方式
    opt = parser.parse_args()
    if opt.local_rank in [-1, 0]:
        print("opt:", opt)

    main(opt)

    
