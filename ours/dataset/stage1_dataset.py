"""
固定取样方式下的数据集
"""
import os
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset as dataset

on_server = True
class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir, pet_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))
        self.pet_list = list(map (lambda x: x.replace('volume', 'pet'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))
        self.pet_list = list(map(lambda x: os.path.join(pet_dir, x), self.pet_list))

    def __getitem__(self, index):
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]
        pet_path = self.pet_list[index]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        pet = sitk.ReadImage(pet_path, sitk.sitkInt16)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)
        pet_array = sitk.GetArrayFromImage(pet)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)
        pet_array = torch.FloatTensor(pet_array).unsqueeze(0)

        return ct_array, seg_array, pet_array

    def __len__(self):
        return len(self.ct_list)


# 第一阶段的训练数据
ct_dir = '/openbayes/input/input0/new_train_ct/'
seg_dir = '/openbayes/input/input0/new_train_seg/'
pet_dir ='/openbayes/input/input0/new_train_pet/'

train_fix_ds = Dataset(ct_dir, seg_dir, pet_dir)
