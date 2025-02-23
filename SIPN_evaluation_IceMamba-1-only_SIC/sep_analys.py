import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def calculate_acc(y_obs, y_sim):
    y_obs = np.transpose(np.array(y_obs), (1, 2, 0))
    y_sim = np.transpose(np.array(y_sim), (1, 2, 0))

    mask = np.load('../data_preprocess/std_mask.npy').astype(bool)
    a = (y_sim - np.mean(y_sim)) * (y_obs - np.mean(y_obs))
    b = np.sqrt((y_sim - np.mean(y_sim))**2) * np.sqrt((y_obs - np.mean(y_obs))**2)
    acc = a / b
    acc = acc[mask, :]
    acc_no_nan = np.mean(acc)
    return acc_no_nan

def calculate_IIEE_single(y_obs, y_sim):
    y_obs_SIE = np.zeros(y_obs.shape)
    y_sim_SIE = np.zeros(y_sim.shape)
    y_obs_SIE[y_obs >= 0.15] = 1
    y_sim_SIE[y_sim >= 0.15] = 1

    union = y_sim_SIE + y_obs_SIE
    union[union == 2] = 1
    intersection = y_sim_SIE * y_obs_SIE
    IIEE_grid = union - intersection
    IIEE = np.sum(IIEE_grid==1) * 625 / 1e6
    return IIEE


# 注意这里，不能对rmse取平均，mse才具有可加性
def calculate_mse_single(y_obs, y_sim):
    mask = np.load('../data_preprocess/std_mask.npy').astype(bool)
    mse = (y_obs - y_sim)**2
    mse = mse[mask]
    mse = np.mean(mse)
    return mse


def calculate_IIEE_year():
    print('IIEE:')
    IIEE_list = []
    for y in range(2001, 2021):
        print(y)
        test = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(y) + '.npz')
        IIEE = []
        for i in range(4):
            IIEE.append(calculate_IIEE_single(y_obs=test['y'], y_sim=test['prd'][:, :, i]))
        IIEE_list.append(IIEE)
        print(IIEE)

    IIEE_list = np.array(IIEE_list)
    mean_IIEE = np.mean(IIEE_list, axis=0)

    print('IIEE mean:')
    print(mean_IIEE)
    print(np.mean(mean_IIEE))


def calculate_rmse_year():
    print('RMSE:')
    mse_list = []
    for y in range(2001, 2021):
        print(y)
        test = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(y) + '.npz')
        mse = []
        for i in range(4):
            mse.append(calculate_mse_single(y_obs=test['y'], y_sim=test['prd'][:, :, i]))
        mse_list.append(mse)
        print(np.sqrt(mse))

    mse_list = np.array(mse_list)
    # 对 mse 取均值，之后在开平方
    mean_mse = np.mean(mse_list, axis=0)
    mean_rmse = np.sqrt(mean_mse)
    print('RMSE mean:')

    print(mean_rmse)
    print(np.mean(mean_rmse))

# 同样的 ACC 也不具有可加性，要完全按照公式计算，不能单独计算后取平均值

def calculate_acc_year():
    acc_list = []
    # 不同 lead time下的预测序列
    prd1 = []
    prd2 = []
    prd3 = []
    prd4 = []
    y = []
    for i in range(2001, 2021):
        test = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(i) + '.npz')
        prd1.append(test['prd'][:, :, 0])
        prd2.append(test['prd'][:, :, 1])
        prd3.append(test['prd'][:, :, 2])
        prd4.append(test['prd'][:, :, 3])
        y.append(test['y'])

    acc_list.append(calculate_acc(y, prd1))
    acc_list.append(calculate_acc(y, prd2))
    acc_list.append(calculate_acc(y, prd3))
    acc_list.append(calculate_acc(y, prd4))

    acc_list = np.array(acc_list)


    print('ACC:')
    print(acc_list)
    print(np.mean(acc_list))


calculate_IIEE_year()
calculate_rmse_year()
calculate_acc_year()
