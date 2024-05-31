import os
import torch
from dataset.dataload_3d import MySet
import numpy as np

'''
This code is to preprocess the nii data of a category into a pth file,
 e.g., assuming that there are 30 AD patient data, 
 i.e., to preprocess the data of these 30 patients and save them uniformly into a pth file.
'''


def nii2pth(d, c, m):
    list = []
    path = '/.../{0}/{1}/{2}'.format(d, c, m)
    for file in os.listdir(path):
        list.append(path + '/' + file)
    label = np.ones(np.shape(list)[0])
    myset = MySet(list, label)
    data_len = myset.__len__()
    tensor, label[0] = myset[0]
    tensor = tensor.numpy()
    tensor = myset.normalize(tensor)
    tensor = torch.from_numpy(tensor)

    for i in range(1, data_len):
        mri_tensor, label[i] = myset[i]
        mri_tensor = mri_tensor.numpy()
        mri_nor = myset.normalize(mri_tensor)
        mri_nor = torch.from_numpy(mri_nor)
        tensor = torch.cat([tensor, mri_nor])

    torch.save(tensor.to(torch.device('cpu')), "/.../{0}/{1}/{2}.pth".format(d, c, m))


if __name__ == '__main__':
    '''
    d: Dataset Name
    c: Category Name
    m: Modality Name
    '''
    d = '...'
    c = '...'
    m = '...'
    path = '/...'
    nii2pth(d, c, m)
