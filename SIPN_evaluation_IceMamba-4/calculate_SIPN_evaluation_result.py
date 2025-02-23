import matplotlib.pyplot as plt
import numpy as np


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


def calculate_mse_single(y_obs, y_sim):
    mask = np.load('../data_preprocess/std_mask.npy').astype(bool)
    mse = (y_obs - y_sim)**2
    mse = mse[mask]
    mse = np.mean(mse)
    return mse


def calculate_IIEE_year():
    IIEE_year = []
    for i in range(2001, 2021):
        result_data = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(i) + '.npz')
        prd = result_data['prd']
        obs = result_data['Y']
        IIEE = []
        for j in range(4):
            single_prd = prd[j, :, :, 3 - j]
            single_obs = obs
            IIEE.append(calculate_IIEE_single(single_obs, single_prd))
        IIEE_year.append(IIEE)
        print(i)
        print(IIEE)

    print('IIEE')
    print(np.mean(IIEE_year, axis=0))
    print(np.mean(IIEE_year))


def calculate_rmse_year():
    mse_year = []
    for i in range(2001, 2021):
        result_data = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(i) + '.npz')
        prd = result_data['prd']
        obs = result_data['Y']
        mse = []
        for j in range(4):
            single_prd = prd[j, :, :, 3 - j]
            single_obs = obs
            mse.append(calculate_mse_single(single_obs, single_prd))
        mse_year.append(mse)
        print(i)
        print(np.sqrt(mse))

    rmse_year = np.sqrt(mse_year)
    print('RMSE:')
    print(np.mean(rmse_year, axis=0))
    print(np.mean(rmse_year))


def calculate_acc_year():
    acc_year = []
    #  注意不能单独计算ACC, 之后再算平均，因为ACC不可加
    # 每个 lead time 的预测结果
    prd1 = []
    prd2 = []
    prd3 = []
    prd4 = []
    y = []

    for i in range(2001, 2021):
        result_data = np.load('./SIPN_evaluation_result/SIPN_prd_' + str(i) + '.npz')
        prd = result_data['prd']
        obs = result_data['Y']
        for j in range(4):
            single_prd = prd[j, :, :, 3 - j]
            if j == 0:
                prd4.append(single_prd)
            elif j == 1:
                prd3.append(single_prd)
            elif j == 2:
                prd2.append(single_prd)
            elif j == 3:
                prd1.append(single_prd)
        # 各lead time 对应y都一样
        y.append(obs)

    acc_year.append(calculate_acc(y, prd4))
    acc_year.append(calculate_acc(y, prd3))
    acc_year.append(calculate_acc(y, prd2))
    acc_year.append(calculate_acc(y, prd1))

    print('ACC:')
    print(acc_year)
    print(np.mean(acc_year, axis=0))


calculate_IIEE_year()
calculate_rmse_year()
calculate_acc_year()