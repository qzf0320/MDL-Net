import shap
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pyspark as pyp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

c1 = 'AD'
c2 = 'MCI'
ii = 1

for ii in [1, 3, 5, 7, 9]:
    plt.rc('font', family='Times New Roman')
    roi = np.load('/.../{0}_vs._{1}/roi(i={2}).npy'.format(
        c1, c2, ii))
    feature = np.load(
        '/.../{0}_vs._{1}/feature(i={2}).npy'.format(
            c1, c2, ii))
    # label = np.load('/home/ubuntu/qiuzifeng/dataset/multi-site/label_adni_adcn1.npy')
    label = np.load(
        '/.../{0}_vs._{1}/label(i={2}).npy'.format(
            c1, c2, ii))
    pca = PCA(n_components=1)
    feature_pca = pca.fit_transform(feature)
    # roi = np.concatenate((roi, feature_pca), axis=1)
    model = xgb.XGBClassifier()
    # model = lgb.LGBMClassifier()
    # model = cb.CatBoostClassifier()
    model.fit(roi, label)
    explainer = shap.TreeExplainer(model)  # 初始化解释器
    shap.initjs()  # 初始化JS
    shap_values = explainer.shap_values(roi)  # 计算每个样本的每个特征的SHAP值
    # shap_values = explainer.shap_interaction_values(roi)  # 计算每个样本的每个特征的SHAP值
    # shap_values = explainer(roi) #计算每个样本的每个特征的SHAP值
    print(shap_values)
    # shap.plots.heatmap(shap_values)
    # shap.bar_plot(shap_values[1], roi[1])
    # shap.force_plot(explainer.expected_value, shap_values, roi) #3860为样本在数据集中的索引
    xlabels = ['PreCG.L', 'PreCG.R', 'SFGdor.L', 'SFGdor.R', 'ORBsup.L', 'ORBsup.R', 'MFG.L', 'MFG.R', 'ORBmid.L',
               'ORBmid.R', 'IFGoperc.L', 'IFGoperc.R', 'IFGtriang.L', 'IFGtriang.R', 'ORBinf.L', 'ORBinf.R', 'ROL.L',
               'ROL.R', 'SMA.L', 'SMA.R', 'OLF.L', 'OLF.R', 'SFGmed.L', 'SFGmed.R', 'ORBsupmed.L', 'ORBsupmed.R',
               'REC.L',
               'REC.R', 'INS.L', 'INS.R', 'ACG.L', 'ACG.R', 'DCG.L', 'DCG.R', 'PCG.L', 'PCG.R', 'HIP.L', 'HIP.R',
               'PHG.L',
               'PHG.R', 'AMYG.L', 'AMYG.R', 'CAL.L', 'CAL.R', 'CUN.L', 'CUN.R', 'LING.L', 'LING.R', 'SOG.L', 'SOG.R',
               'MOG.L', 'MOG.R', 'IOG.L', 'IOG.R', 'FFG.L', 'FFG.R', 'PoCG.L', 'PoCG.R', 'SPG.L', 'SPG.R', 'IPL.L',
               'IPL.R',
               'SMG.L', 'SMG.R', 'ANG.L', 'ANG.R', 'PCUN.L', 'PCUN.R', 'PCL.L', 'PCL.R', 'CAU.L', 'CAU.R', 'PUT.L',
               'PUT.R',
               'PAL.L', 'PAL.R', 'THA.L', 'THA.R', 'HES.L', 'HES.R', 'STG.L', 'STG.R', 'TPOsup.L', 'TPOsup.R', 'MTG.L',
               'MTG.R', 'TPOmid.L', 'TPOmid.R', 'ITG.L', 'ITG.R', 'PCA']
    shap.summary_plot(shap_values, roi, max_display=5, feature_names=xlabels, cmap='cool')
    # plt.savefig('/home/ubuntu/qiuzifeng/dataset/multi-site/ALL/shap/3/{0}_vs._{1}/shap(i={2})'.format(c1, c2, ii))
    # plt.close()
