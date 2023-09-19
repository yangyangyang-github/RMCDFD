import torch
from torch import nn
from d2l import torch as d2l
from net_2 import resnet50, Combine, Encoder  # 目前只用了resnet18
import os
import pandas as pd
from combineModelutils import generate_map1, seedVIG_Datasets1

from torch.utils.data import DataLoader
import pandas as pd
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


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
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
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


def evaluate_accuracy_gpu(net, encoder, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    # net.eval()
    metric = d2l.Accumulator(2)
    correct = 0
    count = 0
    with torch.no_grad():
        for X, y in data_iter:
            X = torch.as_tensor(X, dtype=torch.float).to(device)

            y = y.to(device)
            y_hat, states = net(X)
            y_hat = encoder(y_hat)
            loss = nn.MSELoss()
            y_hat = y_hat.view(-1, 1)
            y = y.view(y_hat.shape)
            metric.add(loss(y_hat, y) * X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
    return metric[0] / metric[1]


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def get_classify_label(Perclos):
    tired_threshold = 0.35
    drowsy_threshold = 0.7
    classify_label = np.repeat(2, Perclos.shape)
    awake_ind = Perclos <= tired_threshold
    classify_label[awake_ind] = 1
    drowsy_ind = Perclos >= drowsy_threshold
    classify_label[drowsy_ind] = 3
    return classify_label


if __name__ == '__main__':
    batch_size = 32
    rnn_hidden_size = 64
    lr, num_epochs = 0.001, 100

    # # # data1 folder for SEED datasets
    label_dir1 = r'/home/yuandy/data/Multichannel_17/CSVPerclos'
    data_dir1 = r'/home/yuandy/data/Multichannel_17/FIR128Randint'
    test_participants = ['s00']
    generate_map1(data_dir1, label_dir1, test_participants)
    train_map1 = r"/home/yuandy/code/self-super/nihe/crossdata_acc/randint_mapfiles1/train_data_map.csv"
    test_map1 = r"/home/yuandy/code/self-super/nihe/crossdata_acc/randint_mapfiles1/test_data_map.csv"

    device = d2l.try_gpu()
    print('training on', device)
    CNNnet = resnet50(classification=False)
    CombNet = Combine(CNNnet, input_size=384 * 4, device=device, batch_first=True)
    CombNet.load_state_dict(torch.load('/home/yuandy/code/self-super/nihe/crossdata_acc/1cnnbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
    CombNet.train()
    CombNet.to(device)
    encoder = Encoder()
    encoder.load_state_dict(torch.load('/home/yuandy/code/self-super/nihe/crossdata_acc/1classbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
    encoder.to(device)
    encoder.train()
    optimizer1 = torch.optim.SGD(CombNet.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    lr_schduler1 = CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=0.05)  # default =0.07
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1, total_epoch=10, after_scheduler=lr_schduler1)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    lr_schduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.05)  # default =0.07
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)

    loss = nn.MSELoss()
    test_l_best = float('inf')
    MMD = MMDLoss()
    train_l = torch.zeros(num_epochs)
    test_l = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        start_num = random.randrange(0, 1024)
        sample_num = 8
        time_num = random.randrange(0, sample_num)
        dataset_type = 'regression'
        train_dataset = seedVIG_Datasets1(train_map1, start_num, time_num, sample_num, dataset_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_set = seedVIG_Datasets1(test_map1, start_num, time_num, sample_num, dataset_type)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)

        data_loader = {"source_loader": train_loader, "target_loader": test_loader}
        print(f'train epoch {epoch}')
        metric = d2l.Accumulator(2)
        CombNet.train()
        encoder.train()
        for X, y in data_loader["source_loader"]:
            optimizer.zero_grad()
            optimizer1.zero_grad()
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            X = X.to(device)
            y = y.to(device)

            y_hat, states = CombNet(X)
            y_hat = encoder(y_hat)

            y_hat = y_hat.view(-1, 1)
            y = y.view(y_hat.shape)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            optimizer1.step()

            with torch.no_grad():
                metric.add(l * X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])

        lr_schduler.step()
        scheduler_warmup.step()
        lr_schduler1.step()
        scheduler_warmup1.step()
        train_l[epoch] = metric[0] / metric[1]
        test_l[epoch] = evaluate_accuracy_gpu(CombNet, encoder, data_loader["target_loader"], device=device)
        print(f'train Loss {train_l[epoch]}  'f'test Loss {test_l[epoch]}')
        if test_l[epoch] < test_l_best:
            test_l_best = test_l[epoch]
            torch.save(encoder.state_dict(), '8_2_1classbest_combRes50_randint_lr1_bs32_acrossSub_run2.params')
            torch.save(CombNet.state_dict(), '8_2_1cnnbest_combRes50_randint_lr1_bs32_acrossSub_run2.params')
    torch.save([train_l, test_l], '8_2_1loss_combRes50_randint_lr1_bs32_acrossSub_run2.data1')
    torch.save(CombNet.state_dict(), '8_2_1end_combRes50_randint_lr1_bs32_acrossSub_run2.params')

    start_num = 0
    time_num = 0
    dataset_type = 'regression'
    sample_num = 1
    train_dataset = seedVIG_Datasets1(train_map1, start_num, time_num, sample_num, dataset_type)
    batch_size = 1
    test_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    Y_label = []
    Y_hat = []
    with torch.no_grad():
        CNNnet = resnet50(classification=False)
        CombNet = Combine(CNNnet, input_size=384 * 4, device=device, batch_first=True)
        CombNet.load_state_dict(torch.load('/home/yuandy/code/self-super/nihe/crossdata_acc/8_2_1cnnbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
        CombNet.eval()
        CombNet.to(device)
        encoder = Encoder()
        encoder.load_state_dict(torch.load('/home/yuandy/code/self-super/nihe/crossdata_acc/8_2_1classbest_combRes50_randint_lr1_bs32_acrossSub_run2.params'))
        encoder.to(device)
        encoder.eval()

        state_size = (1, batch_size, 64)
        init_h = torch.zeros(state_size).to(device)
        # init_c = torch.zeros(state_size).to(device='cuda')
        prev_states = init_h
        for X, y in test_iter:
            X = torch.as_tensor(X, dtype=torch.float).to(device)
            y = y.reshape(-1)
            Y_label += y

            y_hat, prev_states = CombNet(X, prev_states)
            y_hat = encoder(y_hat)
            y_hat = y_hat.reshape(-1)
            Y_hat += y_hat

        Y_label = torch.tensor(Y_label)
        Y_hat = torch.tensor(Y_hat)
        L1_loss = torch.nn.L1Loss()
        L2_loss = torch.nn.MSELoss()
        print(L2_loss(Y_label, Y_hat))
        print(L1_loss(Y_label, Y_hat))
        print(sum(get_classify_label(Y_label) == get_classify_label(Y_hat)) / Y_label.shape)
        predict_cm = confusion_matrix(get_classify_label(Y_label), get_classify_label(Y_hat), labels=[1, 2, 3])
        print(predict_cm)

        # torch.save([Y_label,Y_hat],'regressionResult_randint_int0_900_acrossSub_p03.data1')
        x = torch.arange(Y_label.__len__())
        plt.plot(x, Y_label, label='Perclos')
        plt.plot(x, Y_hat, label='Predict')
        plt.legend(fontsize=12)
        plt.savefig('8_2_1.eps', dpi=400, bbox_inches='tight')
        plt.figure(dpi=400)

        conf_matrix = np.array(predict_cm)
        corrects = predict_cm.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
        per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数
        per_kinds_true = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数
        jing = corrects / per_kinds_true
        zhao = corrects / per_kinds
        print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), train_dataset.__len__()))
        # print(conf_matrix)
        print("每种疲劳总个数：", per_kinds)
        print("每种疲劳预测正确的个数：", corrects)
        print("每种疲劳的识别精确率为：{0}".format([rate * 100 for rate in jing]))
        print("每种疲劳的识别召回率为：{0}".format([rate * 100 for rate in zhao]))
        print("每种疲劳的识别F1为：{0}".format([rate * 100 for rate in jing * zhao * 2 / (jing + zhao)]))

        print("rmse:", sqrt(mean_squared_error(Y_label, Y_hat)))
        Y_label_ave = Y_label.mean()
        Y_hat_ave = Y_hat.mean()
        fenzi = ((Y_label - Y_label_ave) * (Y_hat - Y_hat_ave)).sum()
        fenmu1 = ((Y_label - Y_label_ave) * (Y_label - Y_label_ave)).sum()
        fenmu2 = ((Y_hat - Y_hat_ave) * (Y_hat - Y_hat_ave)).sum()
        fenmu = sqrt(fenmu1 * fenmu2)
        cor = fenzi / fenmu
        print("cor:", cor)