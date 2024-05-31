import torch
import numpy as np
import torch.nn.functional as F
from numpy import random


def load_pth(class1_path, class2_path):
    class1 = torch.load(class1_path)
    class2 = torch.load(class2_path)
    data = torch.cat([class1, class2])
    label1 = np.ones(class1.shape[0])
    label2 = np.zeros(class2.shape[0])
    label = np.hstack((label1, label2))
    label = np.array(label)

    return data, label


def load_roi(roi_class1_path, roi_class2_path):
    roi_class1 = np.load(roi_class1_path)
    roi_class2 = np.load(roi_class2_path)
    roi_data = np.concatenate((roi_class1, roi_class2), axis=0)

    return roi_data


def load_roi_dataset(d, c1, c2):
    global roi
    for i in range(len(d)):
        roi_class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c1)
        roi_class2_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c2)
        roi_data = load_roi(roi_class1_path, roi_class2_path)
        if i == 0:
            roi = roi_data
        else:
            roi = np.concatenate([roi, roi_data], axis=0)

    return roi


def load_dataset_single(d, c1, c2, h, m):
    global roi, label, dataset_out, dataset, dataset_single
    for i in range(len(d)):
        roi_class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c1)
        roi_class2_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c2)
        roi_data = load_roi(roi_class1_path, roi_class2_path)
        if i == 0:
            roi = roi_data
        else:
            roi = np.concatenate([roi, roi_data], axis=0)
    dataset = torch.ones(3, roi.shape[0], 1, 100, 120, 100)

    for i in range(len(d)):
        class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{2}{3}.pth'.format(d[i], c1, h, m)
        class2_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{2}{3}.pth'.format(d[i], c2, h, m)
        data, label_s = load_pth(class1_path, class2_path)

        if i == 0:
            dataset_single = data
            label = label_s
        else:
            dataset_single = torch.cat((dataset_single, data), dim=0)
            label = np.concatenate((label, label_s), axis=0)
    dataset = dataset_single


    dataset = dataset.numpy()
    seed = int(0)

    random.shuffle(dataset)
    random.seed(seed)
    random.shuffle(roi)
    random.seed(seed)
    random.shuffle(label)

    dataset = torch.from_numpy(dataset)
    label = torch.from_numpy(label)
    roi = torch.from_numpy(roi)
    roi = roi.to(torch.float32)

    return dataset, label, roi


def load_dataset_c(d, c1, h, m):
    global roi, label, dataset, dataset_single
    roi_class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d, c1)
    roi_data = np.load(roi_class1_path)
    dataset = torch.ones(3, roi_data.shape[0], 1, 100, 120, 100)
    for idx, j in enumerate(m):
        class_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{2}{3}.pth'.format(d, c1, h, j)
        dataset_single = torch.load(class_path)
        label = np.ones(dataset_single.shape[0])
        dataset[idx] = dataset_single

    label = torch.from_numpy(label)

    for idx, i in enumerate(m):
        if idx == 0:
            dataset_out = torch.cat((dataset[idx], dataset[idx + 1]), dim=1)
        elif 0 < idx < len(m) - 1:
            dataset_out = torch.cat((dataset_out, dataset[idx + 1]), dim=1)

    return dataset_out, label


def load_dataset(d, c1, c2, h, m):
    global roi, label, dataset_out, dataset, dataset_single
    for i in range(len(d)):
        roi_class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c1)
        roi_class2_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{0}_{1}_AAL90.npy'.format(d[i], c2)
        roi_data = load_roi(roi_class1_path, roi_class2_path)
        if i == 0:
            roi = roi_data
        else:
            roi = np.concatenate([roi, roi_data], axis=0)
    dataset = torch.ones(3, roi.shape[0], 1, 100, 120, 100)

    for idx, j in enumerate(m):
        for i in range(len(d)):
            class1_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{2}{3}.pth'.format(d[i], c1, h, j)
            class2_path = '/home/ubuntu/qiuzifeng/dataset/multi-site/{0}/{1}/{2}{3}.pth'.format(d[i], c2, h, j)
            data, label_s = load_pth(class1_path, class2_path)

            if i == 0:
                dataset_single = data
                label = label_s
            else:
                dataset_single = torch.cat((dataset_single, data), dim=0)
                label = np.concatenate((label, label_s), axis=0)
        dataset[idx] = dataset_single

    dataset = dataset.numpy()
    seed = int(0)

    for idx, j in enumerate(m):
        random.seed(seed)
        random.shuffle(dataset[idx])

    random.seed(seed)
    random.shuffle(roi)
    random.seed(seed)
    random.shuffle(label)

    dataset = torch.from_numpy(dataset)
    label = torch.from_numpy(label)
    roi = torch.from_numpy(roi)
    roi = roi.to(torch.float32)

    for idx, i in enumerate(m):
        if idx == 0:
            dataset_out = torch.cat((dataset[idx], dataset[idx + 1]), dim=1)
        elif 0 < idx < len(m) - 1:
            dataset_out = torch.cat((dataset_out, dataset[idx + 1]), dim=1)

    return dataset_out, label, roi


if __name__ == '__main__':
    '''
    d: Dataset Name, e.g., ADNI1, ADNI2
    c1: category1
    c2: category2
    h: File name prefix
    m: modality name
    assume d is ADNI1, c1 is AD, c2 is CN, h is data, and m is GM, that the path is 
    /.../ADNI1/AD/dataGM
    /.../ADNI1/CN/dataGM
    '''
    dataset, label, roi_data = load_dataset(d=['ADNI1', 'ADNI2', 'ADNI3'], c1='AD', c2='CN', h='cropped',
                                            m=['GM', 'WM', 'PET'])

    print(dataset.shape)
    print(label.shape)
    print(roi.shape)
