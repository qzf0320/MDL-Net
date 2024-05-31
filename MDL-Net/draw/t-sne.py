import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import scipy.io as scio
from sklearn import manifold
from sklearn.model_selection import KFold
from torch import einsum
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def t_sne_3d(x, label, c1, c2, i):
    tsne = manifold.TSNE(n_components=3)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    fig = plt.figure()
    axe = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(axe)
    True_labels = label.reshape((-1, 1))

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'z': S_data[:, 2], 'label': S_data[:, 3]})

    colors = ['#FF7C5C', 'palevioletred']
    marker = ['.', '.']
    for index in range(2):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        Z = S_data.loc[S_data['label'] == index]['z']
        axe.scatter(X, Y, Z, cmap='cool', s=40, marker=marker[index], c=colors[index])

    axe.axes.xaxis.set_ticklabels([])
    axe.axes.yaxis.set_ticklabels([])
    axe.axes.zaxis.set_ticklabels([])
    plt.show()
    #
    # plt.title('1', fontsize=32, fontweight='normal', pad=20)
    # plt.savefig(
    #     '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/{0}_vs._{1}/t+f+s1(i={2}, e=200).jpg'.format(c1, c2, i),
    #     dpi=500)


def t_sne(x, label, c1, c2, i):
    tsne = manifold.TSNE(n_components=2)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    True_labels = label.reshape((-1, 1))

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    plt.rc('font', family='Times New Roman')
    colors = ['#8e6fad', '#3e9d35']
    l = ['{}'.format(c2), '{}'.format(c1)]
    marker = ['o', 'o']
    for index in range(2):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=3, c=colors[index])

    plt.legend(l, prop={'size': 12})
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.get_current_fig_manager().window.showMaximized()
    # plt.show()
    plt.savefig(
        '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/{}_vs._{}/ADNI(i={}).jpg'.format(c1, c2, i),
        dpi=600)


def t_sne_multiclass(x, label, i):
    tsne = manifold.TSNE(n_components=2)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    True_labels = label.reshape((-1, 1))
    l = ['CN', 'MCI', 'AD']
    # colors = ['#cf2f2f', '#8e6fad', '#3e9d35']

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    colors = ['#cf2f2f', '#8e6fad', '#3e9d35']
    for index in range(3):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=10, marker='.', c=colors[index])

    plt.legend(l, prop={'size': 8})
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    # plt.show()
    plt.savefig(
        '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/AD_vs._MCI_vs._CN/Net3(i={0}, e=150).jpg'.format(i),
        dpi=500)


if __name__ == '__main__':
    c1 = 'AD'
    c2 = 'MCI'
    for i in [1, 3, 5, 7, 9]:
        x = np.load(
            '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/feature/{}_vs._{}/feature_ADNI(i={}).npy'.format(
                c1, c2, i))
        y = np.load(
            '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/feature/{}_vs._{}/label_ADNI(i={}).npy'.format(
                c1, c2, i))
        t_sne(x, y, c1, c2, i)

    # for i in [1, 3, 5]:
    #     x = np.load(
    #         '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/feature/AD_vs._MCI_vs._CN/feature_Net3(i={}, e=150).npy'.format(
    #             i))
    #     y = np.load(
    #         '/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/t-sne/3/feature/AD_vs._MCI_vs._CN/label_Net3(i={}, e=150).npy'.format(
    #             i))
    #     t_sne_multiclass(x, y, i)
