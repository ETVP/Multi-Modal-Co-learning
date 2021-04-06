from time import time
import os
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss
from net.YNet_kernel3_dropout import net
from dataset.stage1_dataset import train_fix_ds

if not os.path.exists(r'/openbayes/home/Module/'):
    os.makedirs(r'/openbayes/home/Module/')

cudnn.benchmark = True
Epoch = 1000
leaing_rate_base = 1e-5

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
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [400])

# 训练网络
start = time()

for epoch in range(Epoch):

    lr_decay.step()

    loss_step = []

    for step, (ct, seg, pet) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()
        pet = pet.cuda()

        # 如果一个正样本都没有就直接结束
        if torch.numel(seg[seg > 0]) == 0:
            continue

        ct = Variable(ct)
        seg = Variable(seg)
        pet = Variable(pet)

        outputs, outputs_temp = net(ct, pet)
        loss = loss_func(outputs, outputs_temp, seg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # xzl
        loss_step.append(loss.item())

    loss_mean = (sum(loss_step)/len(loss_step))

    print(epoch,loss_mean)

        # if step % 20 is 0:
        #     print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
        #           .format(epoch, step, loss.item(), (time() - start) / 60))

    # 每十个个epoch保存一次模型参数
    if epoch >= 150:
        torch.save(net.state_dict(), '/openbayes/home/Module/net{}.pth'.format(epoch))
