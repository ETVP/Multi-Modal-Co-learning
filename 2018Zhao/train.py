from time import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss
from net.VNet_FCN_Zhao import net
from dataset.stage1_dataset import train_fix_ds

Epoch = 500
leaing_rate_base = 1e-4

batch_size = 1
num_workers = 2
pin_memory = True

net = torch.nn.DataParallel(net).cuda()

# 定义数据加载
train_dl = DataLoader(train_fix_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [400, 450])

# 训练网络
start = time()

for epoch in range(Epoch):

    lr_decay.step()

    loss_step = []

    for step, (ct, pet, seg) in enumerate(train_dl):

        ct = ct.cuda()
        pet = pet.cuda()
        seg = seg.cuda()

        # 如果一个正样本都没有就直接结束
        if torch.numel(seg[seg > 0]) == 0:
            continue

        ct = Variable(ct)
        pet = Variable(pet)
        seg = Variable(seg)

        ct_outputs, pet_outputs, fusion_outputs = net(ct, pet)
        loss = loss_func(ct_outputs, pet_outputs, fusion_outputs, seg)

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_step.append(loss.item())

    loss_mean = (sum(loss_step)/len(loss_step))
    print(epoch,loss_mean)

    # 每十个个epoch保存一次模型参数
    if epoch > 100 and epoch % 2==0:
        torch.save(net.state_dict(), '/openbayes/home/Module/net{}.pth'.format(epoch))
