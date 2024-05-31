""""
we define the data set and data operation in this file.
"""
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import random
import torch
import glob
import os
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class DataList(object):
    def __init__(self, class_path1, class_path2):
        self.class1 = class_path1
        self.class2 = class_path2

    def get_data_list(self):
        class1_mri_list = []
        class2_mri_list = []

        for file in os.listdir(self.class1):
            class1_mri_list.append(self.class1 + '/' + file)

        for file in os.listdir(self.class2):
            class2_mri_list.append(self.class2 + '/' + file)

        class1_label = np.ones(np.shape(class1_mri_list)[0])
        class2_label = np.zeros(np.shape(class2_mri_list)[0])

        mri_list = np.hstack((class1_mri_list, class2_mri_list))
        # mri_list.sort()
        label = np.hstack((class1_label, class2_label))

        return mri_list, label


class MySet(Dataset):

    def __init__(self, data_list1, label):
        self.mri_path = data_list1
        self.label = label

    def __getitem__(self, item):
        mri = self.mri_path[item]
        label = self.label[item]

        mri = sitk.ReadImage(mri)
        mri = sitk.GetArrayFromImage(mri)
        # mri = self.normalize(mri)
        mri = mri[np.newaxis, :, :, :]
        mri = mri[np.newaxis, :, :, :]
        mri_tensor = torch.from_numpy(mri)

        label = np.array(label)
        label_tensor = torch.from_numpy(label)

        return mri_tensor, label_tensor
        # return mri, pet, label

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        # data[data < 0] = 0
        data = np.abs(data)
        # mean = np.mean(data, axis=(2, 3, 4))
        # std = np.std(data, axis=(2, 3, 4))
        # data = (data - mean[:, :, np.newaxis, np.newaxis, np.newaxis])
        # data = data / std[:, :, np.newaxis, np.newaxis, np.newaxis]
        data = (data - np.min(data))/(np.max(data) - np.min(data) + 1e-6)
        return data

    def __len__(self):
        return len(self.mri_path)


if __name__ == '__main__':
    path = ['/home/ubuntu/qiuzifeng/CMFFN/dataset/cropped_AD_GM/m0wrp1ZhouMeiZhu20180823_195609AAHeadScouts001a1001.nii']
    label = np.ones(1)
    myset = MySet(path, label)
    data_len = myset.__len__()
    tensor, label = myset[0]
    tensor = tensor.numpy()
    mri_flip = myset.normalize(tensor)
    mri_flip = torch.from_numpy(mri_flip)
    print(mri_flip.shape)

    torch.save(mri_flip.to(torch.device('cpu')), "myTensor.pth")

    y = torch.load("myTensor.pth")
    print(y)