"""
与kernel3版本的区别在于在参数量最大的六层增加了dropout防止过拟合，效果提升很小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ZhaoNet(nn.Module):
    def __init__(self, training):
        super().__init__()
        self.training = training

        ## PET-CT编码器模块
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16)
        )
        self.encoder_stage11 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16)
        )
        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 64, 2, 2),
            nn.ReLU(64)
        )
        self.down_conv11 = nn.Sequential(
            nn.Conv3d(16, 64, 2, 2),
            nn.ReLU(64)
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
        )
        self.encoder_stage22 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.ReLU(128),
        )
        self.down_conv22 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.ReLU(128),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
        )
        self.encoder_stage33 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.ReLU(128),
        )
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.ReLU(64),
        )
        self.up_conv11 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.ReLU(64),
        )

        # PET-CT解码器模块
        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(64+64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
        )
        self.decoder_stage11 = nn.Sequential(
            nn.Conv3d(64+64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.ReLU(64),
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, 2, 2),
            nn.ReLU(16),
        )
        self.up_conv22 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, 2, 2),
            nn.ReLU(16),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(16+16, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16),
        )
        self.decoder_stage22 = nn.Sequential(
            nn.Conv3d(16+16, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.ReLU(16),
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.ReLU(1),
        )
        self.map11 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.ReLU(1),
        )

        ## 特征融合模块
        self.Fusion = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.ReLU(16),
            nn.Conv3d(16, 64, 3, 1, padding=1),
            nn.ReLU(64),
            nn.Conv3d(64, 256, 3, 1, padding=1),
            nn.ReLU(256),
            nn.Conv3d(256, 1, 1, 1),
            nn.Softmax(),
        )

    def forward(self, inputs, inputs_pet):
        long_range1 = self.encoder_stage1(inputs) + inputs
        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1)
        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2)
        short_range3 = self.up_conv1(long_range3)
        ct_outputs = self.decoder_stage1(torch.cat([short_range3, long_range2], dim=1))
        short_range4 = self.up_conv2(ct_outputs)
        ct_outputs = self.decoder_stage2(torch.cat([short_range4, long_range1], dim=1))
        ct_outputs = F.dropout(ct_outputs, 0.3, self.training)
        ct_outputs = self.map1(ct_outputs)

        long_range11 = self.encoder_stage11(inputs_pet)
        short_range11 = self.down_conv11(long_range11)
        long_range22 = self.encoder_stage22(short_range11)
        short_range22 = self.down_conv22(long_range22)
        long_range33 = self.encoder_stage33(short_range22)
        short_range33 = self.up_conv11(long_range33)
        pet_outputs = self.decoder_stage11(torch.cat([short_range33, long_range22], dim=1))
        short_range44 = self.up_conv22(pet_outputs)
        pet_outputs = self.decoder_stage22(torch.cat([short_range44, long_range11], dim=1))
        pet_outputs = F.dropout(pet_outputs, 0.2, self.training)
        pet_outputs = self.map1(pet_outputs)

        fusion_outputs = self.Fusion(ct_outputs + pet_outputs)
        return ct_outputs, pet_outputs, fusion_outputs

def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)

net = ZhaoNet(training=True)
net.apply(init)

