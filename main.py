import torch
from torch import nn
from d2l import torch as d2l
from net_2 import resnet50, Combine  # 目前只用了resnet18
import os
import pandas as pd
from combineModelutils import generate_map1, seedVIG_Datasets1
from combineModelutils_1 import generate_map, seedVIG_Datasets

from torch.utils.data import DataLoader
import pandas as pd
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # total0 = total
        # total1 = total
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    # net.eval()
    metric = d2l.Accumulator(2)
    correct = 0
    count=0
    with torch.no_grad():
        for X, y in data_iter:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            
            y = y.to(device)            
            y_hat, states = net(X)
            y = y.type(torch.LongTensor).to(device)
            y = y.view(y.shape[0] * y.shape[1])
            y_hat = y_hat.view(-1, 6)
            loss = nn.CrossEntropyLoss()
            metric.add(loss(y_hat, y)*X.shape[0]*X.shape[1], X.shape[0]*X.shape[1])

            _, pred = torch.max(y_hat, dim=1)
            correct += pred.eq(y.data.view_as(pred)).sum()
            count += pred.size(0)
        testaccuracy = float(correct) / count
    return metric[0] / metric[1],testaccuracy

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            
      
if __name__ == '__main__':
    writer = SummaryWriter("/home/yuandy/code/crossdata_acc/logs")

    batch_size = 32
    rnn_hidden_size = 64
    lr, num_epochs = 0.001, 100

    # # # data1 folder for SEED datasets
    label_dir1 = r'/home/yuandy/data/Multichannel_17/CSVPerclos'
    data_dir1 = r'/home/yuandy/data/Multichannel_17/FIR128Randint'
    test_participants = ['s00']
    generate_map1(data_dir1, label_dir1, test_participants)
    train_map1 = r"/home/yuandy/code/crossdata_acc/randint_mapfiles1/train_data_map.csv"
    test_map1 = r"/home/yuandy/code/crossdata_acc/randint_mapfiles1/test_data_map.csv"


    label_dir = r'/home/yuandy/nj/EEG_segments_2/CSVPerclos'
    data_dir = r'/home/yuandy/nj/EEG_segments_2/CSV128Randint'
    TestSubject = ['0']
    generate_map(data_dir, label_dir, TestSubject)
    train_map2 = r"/home/yuandy/code/crossdata_acc/combine_mapfiles/train_data_map.csv"
    test_map2 = r"/home/yuandy/code/crossdata_acc/combine_mapfiles/test_data_map.csv"


    device = d2l.try_gpu()
    print('training on', device)
    CNNnet = resnet50(classification=False)
    CombNet = Combine(CNNnet, input_size=384 * 4, device=device, batch_first=True)
    # net.load_state_dict(torch.load('result\\ResNet18_end_randint_int0_900_acrossSub.params'))
    CombNet.apply(init_weights)

    CombNet.to(device)
    optimizer = torch.optim.SGD(CombNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    lr_schduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.05)  # default =0.07
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)

    loss = nn.CrossEntropyLoss()
    MMD = MMDLoss()
    test_acc_best = 0
    train_l = torch.zeros(num_epochs)
    test_l = torch.zeros(num_epochs)
    train_acc = torch.zeros(num_epochs)
    test_acc = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        correct = 0
        count = 0

        # start_num = 0
        start_num = random.randrange(0,1024)
        sample_num = 8
        # time_num = 0
        time_num = random.randrange(0,sample_num)
        # 设置dataset_type，可以选择‘classification’或‘regression’
        dataset_type = 'classification'
        if epoch%2 == 0:
            # print("===1===")
            train_dataset = seedVIG_Datasets1(train_map1, start_num, time_num, sample_num, dataset_type)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_set = seedVIG_Datasets1(test_map1, start_num, time_num, sample_num, dataset_type)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            # print("***2***")
            train_dataset = seedVIG_Datasets(train_map2, start_num, time_num, sample_num, dataset_type)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            test_set = seedVIG_Datasets(test_map2, start_num, time_num, sample_num, dataset_type)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        data_loader = {"source_loader": train_loader, "target_loader": test_loader}
        a_all = 0
        cls_loss_all = 0
        print(f'train epoch {epoch}')
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(2)
        CombNet.train()
        for X, y in data_loader["source_loader"]:
            optimizer.zero_grad()
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            X = X.to(device)
            y = y.to(device)
            
         
            y_hat, states = CombNet(X)
            y = y.type(torch.LongTensor).to(device)
            y = y.view(y.shape[0] * y.shape[1])
            y_hat = y_hat.view(-1, 6)
            cls_lossl = loss(y_hat, y)
            cls_lossl.backward()
            optimizer.step()

            _, pred = torch.max(y_hat, dim=1)
            correct += pred.eq(y.data.view_as(pred)).sum()
            count += pred.size(0)

            with torch.no_grad():
                metric.add(cls_lossl*X.shape[0]*X.shape[1],  X.shape[0]*X.shape[1])

        train_acc[epoch] = float(correct) / count
        lr_schduler.step()
        scheduler_warmup.step()
        train_l[epoch] = metric[0] / metric[1]
        test_l[epoch],test_acc[epoch] = evaluate_accuracy_gpu(CombNet,data_loader["target_loader"], device=device)
        print(f'train Loss {train_l[epoch]}  'f'test Loss {test_l[epoch]}')
        writer.add_scalar("train/mmd", a_all, epoch)
        writer.add_scalar("train/all_loss", train_l[epoch], epoch)
        writer.add_scalar("test/all_loss", test_l[epoch], epoch)
        writer.add_scalar("train/class-loss", cls_loss_all, epoch)
        writer.add_scalar("train/acc", train_acc[epoch], epoch)
        writer.add_scalar("test/acc", test_acc[epoch], epoch)
        if test_acc[epoch] > test_acc_best:
            test_acc_best = test_acc[epoch]
            torch.save(CombNet.state_dict(), 'best_combRes50_randint_lr1_bs32_acrossSub_run2.params')
    torch.save([train_l,test_l],'loss_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
    torch.save([train_acc,test_acc],'acc_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
    torch.save(CombNet.state_dict(), 'end_combRes50_randint_lr1_bs32_acrossSub_run2.params')
