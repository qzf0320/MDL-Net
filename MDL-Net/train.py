import os

from numpy import random
from torch.utils.data import TensorDataset, DataLoader
import torch
from dataset.load_dataset import load_dataset
from model.MDL_Net import generate_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, manifold
from configs import load_config
from sklearn.model_selection import KFold
import scipy.io as scio

count = 1
c1 = 'AD'
c2 = 'MCI'
i = 1


def save_checkpoint(best_acc, model, optimizer, args, epoch):
    global path
    path = '/.../{0}_vs._{1}/i={2}-{3}.pth'.format(
        c1, c2, i, count)
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', path))


# 输入数据推荐使用numpy数组，使用list格式输入会报错
def K_Flod_spilt(K, fold, data, label):
    '''
    :param K: The number of parts into which the dataset is to be divided. If ten times ten folds take K=10
    :param fold: To fetch the data of the first fold. If you want to take the 5th flod then flod=5
    '''
    split_train_list = []
    split_test_list = []
    kf = KFold(n_splits=K)
    for train, test in kf.split(data):
        split_train_list.append(train.tolist())
        split_test_list.append(test.tolist())
    train, test = split_train_list[fold], split_test_list[fold]
    return data[train], data[test], label[train], label[test]  # 已经分好块的数据集


def t_sne(x, label):
    tsne = manifold.TSNE(n_components=2)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    True_labels = label.reshape((-1, 1))

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    colors = ['#8e6fad', '#cf2f2f']
    l = ['{}'.format(c2), '{}'.format(c1)]
    marker = ['o', 'o']
    for index in range(2):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=5, marker=marker[index], c=colors[index])

    plt.legend(l)
    plt.xticks([])
    plt.yticks([])
    #
    # plt.title('1', fontsize=32, fontweight='normal', pad=20)
    plt.savefig('/.../{0}_vs._{1}/t-sne_(i={2}).jpg'.format(c1, c2, i), dpi=500)


def _train(epoch, train_loader, roi_train_loader, model, optimizer, criterion_cls, criterion_roi, args):
    model.train()

    losses = 0.
    losses_cls = 0.
    losses_roi = 0.
    acc = 0.
    total = 0.
    n = 0.
    for idx, ((data, target), (_, roi_train)) in enumerate(zip(train_loader, roi_train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.long().cuda()
            roi_data = roi_train.cuda()
        output, roi_out = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        loss_cls = criterion_cls(output, target)
        loss_roi = 10 * criterion_roi(roi_out, roi_data)
        loss = loss_cls + loss_roi

        losses_cls += loss_cls
        losses_roi += loss_roi
        losses += loss
        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        n = idx
    print(
        '[{0}][Epoch: {1:4d}], Loss: {2:.3f}, Loss_cls: {3:.3f}, Loss_roi: {4:.3f}, Acc: {5:.3f}, Correct {6} / Total {7}'.format(
            count, epoch, losses / (n + 1), losses_cls / (n + 1), losses_roi / (n + 1),
                          acc / total * 100., acc, total))
    r_loss = losses_cls / (n+1)
    r_loss = r_loss.cpu().detach().numpy()
    return r_loss


def _eval(epoch, test_loader, roi_test_loader, model, args):
    model.eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    i = 0.
    losses_ce = 0.
    with torch.no_grad():
        for idx, ((data, target), (_, roi_test)) in enumerate(zip(test_loader, roi_test_loader)):
            if args.cuda:
                data, target = data.cuda(), target.long().cuda()
                roi_data = roi_test.cuda()
            output, roi_out = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)
            for i in range(len(pred)):
                pred_matrix.append(pred[i].cpu())
                target_matrix.append(target[i].cpu())
            acc += pred.eq(target).sum().item()
            matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]
        # matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        precision = TP / (TP + FP)
        Sen = TP / (TP + FN)
        recall = TP / (TP + FN)
        Spe = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        print('[{0}][Epoch: {1:4d}, Loss: {2}]'.format(count, epoch, losses_ce/(idx + 1)))
        print('cls: Acc: {0:.3f}, Sen: {1:.4f}, Spe: {2:.4f}, F1: {3:.4f}, BAC: {4:.4f}'.format(
            acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2))

    return acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2


def _test(epoch, test_loader, roi_test_loader, model, args):
    model.eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    # output_tsne = np.zeros(shape=[1, 2])
    output_tsne = []
    target_tsne = []
    roi_matrix = []
    roi_ground = []
    score = np.zeros(shape=[1, 2])
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    with torch.no_grad():
        for idx, ((data, target), (_, roi_test)) in enumerate(zip(test_loader, roi_test_loader)):
            if args.cuda:
                data, target = data.cuda(), target.long().cuda()
                roi_data = roi_test.cuda()

            output, roi_out, att = model(data)
            if idx == 0:
                output_tsne = output.cpu().numpy()
                target_tsne = target.cpu().numpy()
                roi_matrix = roi_out.cpu().numpy()
                roi_ground = roi_data.cpu().numpy()

            else:
                output_tsne = np.vstack([output_tsne, output.cpu().numpy()])
                target_tsne = np.concatenate([target_tsne, target.cpu().numpy()], axis=0)
                roi_matrix = np.concatenate([roi_matrix, roi_out.cpu().numpy()], axis=0)
                roi_ground = np.concatenate([roi_ground, roi_data.cpu().numpy()], axis=0)

            _, pred = F.softmax(output, dim=-1).max(1)
            out = F.softmax(output, dim=-1)
            if idx == 0:
                score[0][0] = out[0][0].cpu().numpy()
                score[0][1] = out[0][1].cpu().numpy()
            else:
                score = np.vstack([score, out.cpu().numpy()])

            for k in range(len(pred)):
                pred_matrix.append(pred[k].cpu())
                target_matrix.append(target[k].cpu())
            acc += pred.eq(target).sum().item()
            bio_onehot = np.empty(shape=[0, 2])
            # label = label_binarize(target_matrix, classes=np.array(list(range(2))))
            for i, value in enumerate(target_matrix):
                if value == 0:
                    bio_onehot = np.concatenate((bio_onehot, [[1, 0]]), 0)
                if value == 1:
                    bio_onehot = np.concatenate((bio_onehot, [[0, 1]]), 0)
            label = bio_onehot
            matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]

        # t_sne(output_tsne, label.ravel())
        # matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        precision = TP / (TP + FP)
        Sen = TP / (TP + FN)
        recall = TP / (TP + FN)
        Spe = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall)
        fpr, tpr, theresholds = metrics.roc_curve(label.ravel(), score.ravel(), pos_label=1, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        print('[{0}][Epoch: {1:4d}]'.format(count, epoch))
        print('cls: Acc: {0:.3f}, Sen: {1:.4f}, Spe: {2:.4f}, F1: {3:.4f}, BAC: {4:.4f}'.format(
            acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2))

    return acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (
            Sen + Spe) / 2, auc, fpr, tpr, output_tsne, target_tsne, roi_matrix, matrix


def main(args):
    global count, c1, c2, i
    '''load and process dataset'''
    dataset, label, roi_data = load_dataset(d=['AIBL'], c1=c1, c2=c2, h='cropped',
                                            m=['GM', 'WM', 'PET'])
    print(dataset.shape)
    k = 10
    '''split dataset and train'''
    for i in [7]:
        train_loss = []
        test_acc = []
        test_sen = []
        test_spe = []
        test_bac = []
        test_f1 = []
        test_auc = []
        test_fpr = []
        test_tpr = []
        test_feature = []
        test_target = []
        test_roi = []
        test_matrix = []
        count = 1
        for ii in range(k):
            print('iter={0}'.format(i))
            train_data, val_test_data, label_train, label_val_test = K_Flod_spilt(k, ii, dataset, label)
            val_data, test_data, label_val, label_test = K_Flod_spilt(k, ii, val_test_data, label_val_test)
            roi_train, roi_val_test, _, _ = K_Flod_spilt(k, ii, roi_data, label)
            roi_val, roi_test, _, _ = K_Flod_spilt(k, ii, roi_val_test, label_val_test)

            train = TensorDataset(train_data, label_train)
            val = TensorDataset(val_data, label_val)
            test = TensorDataset(test_data, label_test)
            roi_train = TensorDataset(train_data, roi_train)
            roi_val = TensorDataset(val_data, roi_val)
            roi_test = TensorDataset(test_data, roi_test)

            torch.manual_seed(1)
            train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True)
            roi_train_loader = DataLoader(dataset=roi_train, batch_size=args.batch_size, shuffle=True)
            roi_val_loader = DataLoader(dataset=roi_val, batch_size=args.batch_size, shuffle=True)
            roi_test_loader = DataLoader(dataset=roi_test, batch_size=1, shuffle=True)

            print('{0} fold:'.format(ii + 1))

            model = generate_model(model_depth=18, in_planes=1, num_classes=2, iter=i)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  momentum=args.momentum)

            start_epoch = 1

            if args.cuda:
                model = model.cuda()

            criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.2)
            criterion_roi = nn.SmoothL1Loss()
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)

            global_acc = 0.
            global_sen = 0.
            global_spe = 0.
            global_f1 = 0.
            global_bac = 0.
            global_epoch = 0.
            for epoch in range(start_epoch, args.epochs + 1):
                loss = _train(epoch, train_loader, roi_train_loader, model, optimizer, criterion_cls, criterion_roi, args)
                train_loss.append(loss)
                best_acc, best_sen, best_spe, best_f1_score, best_bac = _eval(
                    epoch,
                    val_loader,
                    roi_val_loader,
                    model,
                    args)
                if global_acc < best_acc and best_sen != 0 and best_spe != 0:
                    global_acc = best_acc
                    global_sen = best_sen
                    global_spe = best_spe
                    global_f1 = best_f1_score
                    global_bac = best_bac
                    global_epoch = epoch
                    save_checkpoint(best_acc, model, optimizer, args, epoch)
                elif global_acc == best_acc:
                    if global_bac < best_bac:
                        global_acc = best_acc
                        global_sen = best_sen
                        global_spe = best_spe
                        global_f1 = best_f1_score
                        global_bac = best_bac
                        global_epoch = epoch
                        save_checkpoint(best_acc, model, optimizer, args, epoch)
                    elif global_bac == best_bac:
                        if global_epoch < epoch:
                            global_acc = best_acc
                            global_sen = best_sen
                            global_spe = best_spe
                            global_f1 = best_f1_score
                            global_bac = best_bac
                            global_epoch = epoch
                            save_checkpoint(best_acc, model, optimizer, args, epoch)

                lr_scheduler.step()
                print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
            print(
                '[{0}][Acc: {1:.3f}, Sen: {2:.4f}, Spe: {3:.4f}, F1: {4:.4f}, BAC: {5:.4f}]'.format(
                    ii + 1, global_acc, global_sen, global_spe, global_f1, global_bac))
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            checkpoints = torch.load(os.path.join('checkpoints', path))
            model.load_state_dict(checkpoints['model_state_dict'])
            optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            start_epoch = checkpoints['global_epoch']
            acc, sen, spe, f1_score, bac, auc, fpr, tpr, feature, target, roi_out, matrix = _test(
                start_epoch, test_loader, roi_test_loader, model, args)
            print('The shape of feature:{}'.format(feature.shape))
            print('The shape of target:{}'.format(target.shape))
            print('The shape of roi:{}'.format(roi_out.shape))
            test_acc.append(acc)
            test_sen.append(sen)
            test_spe.append(spe)
            test_f1.append(f1_score)
            test_bac.append(bac)
            test_auc.append(auc)
            test_fpr.append(fpr)
            test_tpr.append(tpr)
            if count == 1:
                test_feature = feature
                test_roi = roi_out
                test_target = target
                test_matrix = matrix
            else:
                test_feature = np.vstack([test_feature, feature])
                test_roi = np.vstack([test_roi, roi_out])
                test_target = np.concatenate([test_target, target], axis=0)
                test_matrix = test_matrix + matrix
            count = count + 1

            np.save('/.../{0}_vs._{1}/loss_train(i={2}, k={3}).npy'.format(c1, c2, i, ii), train_loss)
            train_loss = []

        fpr_path = '/.../{0}_vs._{1}/i={2}_fpr.mat'.format(c1, c2, i)
        tpr_path = '/.../{0}_vs._{1}/i={2}_tpr.mat'.format(c1, c2, i)
        scio.savemat(fpr_path, {'Net_fpr': test_fpr})
        scio.savemat(tpr_path, {'Net_tpr': test_tpr})

        np.save('/.../{0}_vs._{1}/i={2}_fpr.mat'.format(c1, c2, i), test_fpr)
        np.save('/.../{0}_vs._{1}/i={2}_tpr.mat'.format(c1, c2, i), test_tpr)

        print(
            'cls: Acc: {0:.3f}±{1:.3f}, Sen: {2:.4f}±{3:.4f}, Spe: {4:.4f}±{5:.4f}, F1: {6:.4f}±{7:.4f}, BAC: {8:.4f}±{9:.4f}, AUC: {10:.4f}±{11:.4f}'.format(
                np.mean(test_acc), np.std(test_acc), np.mean(test_sen), np.std(test_sen), np.mean(test_spe),
                np.std(test_spe), np.mean(test_f1), np.std(test_f1), np.mean(test_bac), np.std(test_bac),
                np.mean(test_auc), np.std(test_auc)))

        np.save(
            '/.../{0}_vs._{1}/feature_(i={2}).npy'.format(
                c1, c2, i), test_feature)
        np.save(
            '/.../{0}_vs._{1}/label_(i={2}).npy'.format(
                c1, c2, i), test_target)
        t_sne(test_feature, test_target)

        result_cls = {'ACC': test_acc, 'SEN': test_sen, 'SPE': test_spe, 'F1': test_f1, 'BAC': test_bac,
                      'AUC': test_auc}
        np.save('/.../{0}_vs._{1}/cls_result.npy'.format(
            c1, c2), result_cls)

        np.save(
            '/.../{0}_vs._{1}/confusion_matrix_(i={2}).npy'.format(
                c1, c2, i), test_matrix)
        np.save(
            '/.../{0}_vs._{1}/roi(i={2}).npy'.format(c1, c2, i),
            test_roi)



if __name__ == '__main__':
    args = load_config()
    main(args)
