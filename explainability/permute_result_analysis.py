import numpy as np


def calcuate_MAE_all(y_true, y_pred):
    mask = np.load('../data_preprocess/g2202_land.npy')
    mask_y_true = y_true[:, mask.astype(bool), :]
    mask_y_pred = y_pred[:, mask.astype(bool), :]
    mae = np.mean(np.abs(mask_y_true - mask_y_pred))
    return mae


def calcuate_RMSE_all(y_true, y_pred):
    mask = np.load('../data_preprocess/g2202_land.npy')
    mask_y_true = y_true[:, mask.astype(bool), :]
    mask_y_pred = y_pred[:, mask.astype(bool), :]
    rmse = np.sqrt(np.mean((mask_y_true - mask_y_pred) ** 2))
    return rmse


def calculate_acc_signal(y_obs, y_sim):
    # 计算均值
    mean_obs = np.mean(y_obs)
    mean_sim = np.mean(y_sim)

    # 计算协方差
    covariance = np.mean((y_sim - mean_sim) * (y_obs - mean_obs))

    # 计算标准差
    std_obs = np.std(y_obs)
    std_sim = np.std(y_sim)

    # 计算ACC (相关系数)
    if std_obs > 0 and std_sim > 0:  # 避免除以零
        acc = covariance / (std_obs * std_sim)
    else:
        acc = np.nan  # 如果标准差为0，返回NaN

    return acc


def calculate_IIEE_all(y_obs, y_sim):
    SIE_obs = np.zeros_like(y_obs)
    SIE_obs[y_obs >= 0.15] = 1
    SIE_sim = np.zeros_like(y_sim)
    SIE_sim[y_sim >= 0.15] = 1
    union = SIE_sim + SIE_obs
    union[union == 2] = 1
    intersection = SIE_sim * SIE_obs
    IIEE_area = union - intersection
    IIEE = (np.sum(IIEE_area == 1) * 625 / 1e6) / (IIEE_area.shape[0] * IIEE_area.shape[-1])
    return IIEE


def compare_all(variable, seed, original_mae, original_rmse, original_acc, original_IIEE):
    y_true = np.load('../Model/IceMamba-6/y_2015-2022.npy')
    y_pred = np.load('./old/seed_' + str(seed) + '/prd_' + str(variable) + '_seed_' + str(seed) + '.npy')

    mae = calcuate_MAE_all(y_true, y_pred)
    print('MAE:', mae - original_mae)
    rmse = calcuate_RMSE_all(y_true, y_pred)
    print('RMSE:', rmse - original_rmse)
    IIEE = calculate_IIEE_all(y_true, y_pred)
    print('IIEE:', IIEE - original_IIEE)
    acc = calculate_acc_signal(y_true, y_pred)
    print('ACC:', acc - original_acc)
    print('---')


if __name__ == '__main__':
    y_true = np.load('../Model/IceMamba-6/y_2015-2022.npy')
    y_pred = np.load('../Model/IceMamba-6/prd_2015-2022.npy')
    original_mae = calcuate_MAE_all(y_true, y_pred)
    original_rmse = calcuate_RMSE_all(y_true, y_pred)
    original_IIEE = calculate_IIEE_all(y_true, y_pred)
    original_acc = calculate_acc_signal(y_true, y_pred)

    for seed in range(32, 33):
        for variable in range(41):
            print('seed: ', seed)
            print('variable: ', variable)
            compare_all(variable, seed, original_mae, original_rmse, original_acc, original_IIEE)



