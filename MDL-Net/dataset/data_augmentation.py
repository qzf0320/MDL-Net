import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from nibabel.viewers import OrthoSlicer3D
import torchvision
from monai.transforms import AddChannel, Compose, RandAffine, RandRotate90, RandFlip, apply_transform, ToTensor, \
    Orientationd, Flip, RandGaussianNoise

'''
DCE 
'''


def read_img(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img.GetSpacing()


def crop_VOI_aug_Save(Ori_Path, d, c, m):
    for ii, patient_path in enumerate(glob.glob(Ori_Path + '/*.nii')):
        path = {}
        save_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/gaussian{2}/'.format(d, c, m)
        count = patient_path.count('/')
        path = patient_path.partition('/')
        for i in range(count-1):
            path1 = path[2].partition('/')
            path = path1
        data, _ = read_img(patient_path)
        # train_transforms = Compose([AddChannel(), Flip(spatial_axis=1)])
        gaussiannoise = Compose([AddChannel(), RandGaussianNoise()])

        # data_trans = apply_transform(train_transforms, data.astype(np.float))
        # data_trans = data_trans.squeeze(0)
        data_g = apply_transform(gaussiannoise, data.astype(np.float))
        data_g = data_g.squeeze(0)
        # OrthoSlicer3D(data, title='image').show()
        # OrthoSlicer3D(data_trans, title='image_transformed').show()
        # plt.figure(1)
        # plt.subplot(1, 2, 1)
        # plt.title("image")
        # plt.imshow(data[20, :, :], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.title("image_transformed")
        # plt.imshow(data_trans[20, :, :], cmap="gray")
        # plt.show()
        # out = sitk.GetImageFromArray(data_trans)
        out_g = sitk.GetImageFromArray(data_g)
        if not os.path.isdir(save_path):
            # 创建文件夹
            os.mkdir(save_path)
        sitk.WriteImage(out_g, save_path + 'g_{0}.nii'.format(path1[2]))


for dataset in ['ADNI1', 'ADNI2', 'ADNI3']:
    for c in ['AD', 'CN', 'MCI']:
        for m in ['GM', 'WM', 'PET']:
            Ori_Path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/cropped{2}/'.format(dataset, c, m)
            crop_VOI_aug_Save(Ori_Path, dataset, c, m)
