"""
Dice loss
用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ct_pred, pet_pred, fusion_pred, target):

        # dice系数的定义
        ct_pred = ct_pred.squeeze(dim=1)
        ct_dice = 2 * (ct_pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (ct_pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)
        # dice系数的定义
        pet_pred = pet_pred.squeeze(dim=1)
        pet_dice = 2 * (pet_pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pet_pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        fusion_pred = fusion_pred.squeeze(dim=1)
        fusion_dice = 2 * (fusion_pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (fusion_pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        # 返回的是dice距离
        return ((1 - ct_dice).mean() + (1 - pet_dice).mean()) * 0.5 + (1 - fusion_dice).mean()
