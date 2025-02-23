import os
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns


def calculate_RMSE_and_MAE_order_by_month(y_prediction, y_true, file_name, start_date='2016-01'):
    # 保存每个月的误差总值
    mae_all = np.zeros(12)
    # 统计测试集中各个月份的分布
    month_distribution = np.zeros(12)
    # 开始的时间
    pd_start_date = pd.to_datetime(start_date)
    mask = np.load('../data_preprocess/g2202_land.npy')

    # 遍历每一个样本
    for i in range(y_true.shape[0]):
        # 样本对应的起始时间，预测窗口为6个月，step为1个月
        sample_start_date = pd_start_date + pd.DateOffset(months=i)
        # 计算预测窗口的时间
        for j in range(y_true.shape[-1]):
            sample_date = sample_start_date + pd.DateOffset(months=j)
            sample_month = sample_date.month
            y_obs = y_true[i, :, :, j]
            mask_obs = y_obs[mask.astype(bool)]
            y_prd = y_prediction[i, :, :, j]
            mask_prd = y_prd[mask.astype(bool)]
            mae = mean_absolute_error(mask_obs, mask_prd)
            mae_all[sample_month - 1] += mae
            # 统计月份
            month_distribution[sample_month - 1] += 1

    mae_all = mae_all / month_distribution

    path = './result/seed_' + file_name.split('_')[-1] + '_month/'
    if not os.path.exists(path):
        os.makedirs(path)
    # 持久化SIC loss
    np.save(path + file_name, mae_all)
    return mae_all


def calculate_MAE_order_by_lead_time(y_prediction, y_true, file_name):
    mae_list = []
    mask = np.load('../data_preprocess/g2202_land.npy')
    mask = np.repeat(mask[np.newaxis, ...], y_prediction.shape[0], axis=0)
    # lead time 前置时间
    for i in range(y_true.shape[-1]):
        y_obs = y_true[:, :, :, i]
        mask_obs = y_obs[mask.astype(bool)]
        y_prd = y_prediction[:, :, :, i]
        mask_prd = y_prd[mask.astype(bool)]

        mae_list.append(np.mean(np.abs(mask_obs - mask_prd)))

    mae_list = np.array(mae_list)
    path = './result/seed_' + file_name.split('_')[-1] + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # 持久化SIC loss
    np.save(path + file_name, mae_list)
    return mae_list


def _calculate_MAE_seed(variable, seed, y_true):
    y_pred = np.load('./result/seed_' + str(seed) + '/prd_' + str(variable) + '_seed_' + str(seed) + '.npy')
    calculate_MAE_order_by_lead_time(y_pred, y_true, file_name='variable_' + str(variable) + '_seed_' + str(seed))
    calculate_RMSE_and_MAE_order_by_month(y_pred, y_true, file_name='variable_' + str(variable) + '_seed_' + str(seed))


def calculate_MAE_seed():
    y_true = np.load('./test_set_era5_combine_icemamba-6.npz')['Y']
    for seed in range(23, 33):
        for variable in range(50):
            print('seed: ', seed)
            print('variable: ', variable)
            _calculate_MAE_seed(variable, seed, y_true)


def mean_MAE_all_seed():
    save_path = 'result/seed_mean/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for variable in range(50):
        MAE = np.zeros(shape=(6))
        for seed in range(23,  33):
            path = './result/seed_' + str(seed) + '/variable_' + str(variable) + '_seed_' + str(seed) + '.npy'
            MAE += np.load(path)
        # 注意修改
        MAE = MAE / 10

        np.save(save_path + 'variable_' + str(variable) + '_mean_MAE_all_seed.npy', MAE)

    save_path = 'result/seed_mean_month/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for variable in range(50):
        MAE = np.zeros(shape=(12))
        for seed in range(23, 33):
            path = './result/seed_' + str(seed) + '_month/variable_' + str(variable) + '_seed_' + str(seed) + '.npy'
            MAE += np.load(path)
        # 注意修改
        MAE = MAE / 10
        np.save(save_path + 'variable_' + str(variable) + '_mean_MAE_all_seed.npy', MAE)


def mae_permute_lead_time():
    path = 'result/seed_mean_lead_time/'

    original_mae = calculate_MAE_order_by_lead_time(np.load('result/org/prd_org.npy'),
                                                    np.load('./test_set_era5_combine_icemamba-6.npz')['Y'],
                                                    file_name='original_mae.npy')

    if not os.path.exists(path):
        os.mkdir(path)

    # lead time
    for i in range(50):
        MAE = np.zeros(shape=(50, 6))
        for j in range(50):
            mae_permute = np.load('./result/seed_mean/variable_' + str(j) + '_mean_MAE_all_seed' + '.npy') - original_mae
            MAE[j, :] += mae_permute

    MAE_all = MAE * 10000
    # 定义纵轴和横轴的标签
    y_axis_labels = [
        "u10 (3)", "u10 (2)", "u10 (1)",
    ]
    x_axis_labels = [i for i in range(1, 7)]

    # 设置字体大小
    plt.rcParams['axes.labelsize'] = 15  # 坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 15  # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 15  # y轴刻度字体大小
    plt.rcParams['figure.titlesize'] = 15  # 图表标题字体大小
    plt.rcParams['font.size'] = 15  # colorbar字体大小

    # 自定义颜色映射
    colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
              "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    cmap = ListedColormap(colors)

    # 创建热力图
    plt.figure(figsize=(6, 25))
    sns.heatmap(MAE_all, annot=True, fmt=".1f", cmap=cmap, xticklabels=x_axis_labels,
                yticklabels=y_axis_labels, vmax=10, cbar_kws={"shrink": 0.5})  # 取消数字显示

    # 设置热力图的标题和标签
    plt.xlabel('Lead time (month)')
    plt.ylabel('Input variable name')
    plt.title(r'MAE change $ \times 10^{-2}$ %')
    # 显示热力图
    plt.savefig('era5_lead_time.svg', dpi=600, bbox_inches='tight')
    plt.show()


def mae_permute_month():
    path = 'result/seed_mean_lead_time/'
    original_mae = calculate_RMSE_and_MAE_order_by_month(np.load('result/org/prd_org.npy'),
                                                         np.load('./test_set_era5_combine_icemamba-6.npz')['Y'],
                                                         file_name='original_mae.npy')

    if not os.path.exists(path):
        os.mkdir(path)

    # lead time
    for i in range(50):
        MAE = np.zeros(shape=(50, 12))
        for j in range(50):
            mae_permute = np.load('./result/seed_mean_month/variable_' + str(j) + '_mean_MAE_all_seed' + '.npy') - original_mae
            MAE[j, :] += mae_permute

    MAE_all = MAE * 10000
    # 定义纵轴和横轴的标签
    y_axis_labels = [
        "u10 (3)", "u10 (2)", "u10 (1)",
    ]
    x_axis_labels = [i for i in range(1, 13)]

    # 设置字体大小
    plt.rcParams['axes.labelsize'] = 15  # 坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 15  # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 15  # y轴刻度字体大小
    plt.rcParams['figure.titlesize'] = 15  # 图表标题字体大小
    plt.rcParams['font.size'] = 15  # colorbar字体大小

    # 自定义颜色映射
    colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
              "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    cmap = ListedColormap(colors)

    # 创建热力图
    plt.figure(figsize=(12, 25))
    sns.heatmap(MAE_all, annot=True, fmt=".1f", cmap=cmap, xticklabels=x_axis_labels,
                yticklabels=y_axis_labels, vmax=10, cbar_kws={"shrink": 0.5})  # 取消数字显示

    # 设置热力图的标题和标签
    plt.xlabel('Target month')
    plt.ylabel('Input variable name')
    plt.title(r'MAE change $ \times 10^{-2}$ %')
    # 显示热力图
    plt.savefig('era5_month.svg', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    calculate_MAE_seed()
    mean_MAE_all_seed()
    mae_permute_lead_time()
    mae_permute_month()