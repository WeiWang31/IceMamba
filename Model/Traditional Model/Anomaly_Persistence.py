import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


# 预测从2016年1月开始，为了配合test set
def anomaly_persistence_forecast():
    agg_sic = np.load('../../data_preprocess/SIC/g2202_197901_202312.npy')
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0

    for j in range(1, 7):
        persistence = []
        for i in range(agg_sic.shape[-1]):
            if i >= 12 * (2016 - 1979):
                #
                persistence.append(
                    agg_sic[:, :, (i - j)] - np.mean(agg_sic[:, :, (i - j) - 12 * 10:(i - j):12], axis=2) + np.mean(agg_sic[:, :, i - 12 * 10:i:12], axis=2))
        persistence = np.transpose(np.array(persistence), axes=(1, 2, 0))
        np.save('G2202_persistence_forcast_lead_month_' + str(j), persistence[:, :, :])


# 计算某个lead time的损失
def calculate_mae_mse(y_prediction, y_true, start_date='2016-01'):
    # 保存每个月的误差总值
    mse_all = np.zeros(12)
    mae_all = np.zeros(12)
    # 统计测试集中各个月份的分布
    month_distribution = np.zeros(12)
    # 开始的时间
    pd_start_date = pd.to_datetime(start_date)
    mask = np.load('../../data_preprocess/g2202_land.npy')

    # 遍历每一个样本
    for i in range(y_true.shape[-1]):
        # 样本对应的起始时间，预测窗口为6个月，step为1个月
        sample_date = pd_start_date + pd.DateOffset(months=i)
        sample_month = sample_date.month
        mse = mean_squared_error(y_true[mask==1, i], y_prediction[mask==1, i])
        mae = mean_absolute_error(y_true[mask==1, i], y_prediction[mask==1, i])

        mse_all[sample_month - 1] += mse
        mae_all[sample_month - 1] += mae
        # 统计月份
        month_distribution[sample_month - 1] += 1


    mse_all = mse_all / month_distribution
    # rmse_all = np.sqrt(mse_all)
    mae_all = mae_all / month_distribution
    # 持久化SIC loss
    return mae_all, mse_all


# 计算总体预测的损失
def calculate_mae_rmse_persistence():
    agg_sic = np.load('../../data_preprocess/SIC/g2202_197901_202312.npy')
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0
    obs = agg_sic[:, :, 12 * (2016 - 1979):(2023 - 1979) * 12]

    mae_list = []
    mse_list = []

    for i in range(1, 7):
        prd = np.load('G2202_persistence_forcast_lead_month_' + str(i) + '.npy')[:, :, :-12]
        mae, mse = calculate_mae_mse(y_prediction=prd, y_true=obs, start_date='2016-01')
        mae_list.append(mae)
        mse_list.append(mse)

    mse_list = np.array(mse_list)
    mse_list_month = np.mean(mse_list, axis=0)
    mse_list_lead = np.mean(mse_list, axis=1)

    mae_list_month = np.mean(mae_list, axis=0)
    mae_list_lead = np.mean(mae_list, axis=1)

    print('RMSE:')
    print(np.sqrt(np.mean(mse_list)))
    print('MAE:')
    print(np.mean(mae_list))

    np.savez('G2202_persistence_RMSE_MAE_lead_time',
             MAE=mae_list_lead,
             RMSE=np.sqrt(mse_list_lead))

    np.savez('G2202_persistence_RMSE_MAE_month',
             MAE=mae_list_month,
             RMSE=np.sqrt(mse_list_month))

    mae_list = np.array(mae_list)
    rmse_list = np.sqrt(mse_list)

    np.savez('G2202_persistence_RMSE_MAE',
             MAE=mae_list,
             RMSE=rmse_list)


def calculate_IIEE_single(y_obs, y_sim):
    union = y_sim + y_obs
    union[union == 2] = 1
    intersection = y_sim * y_obs
    IIEE_area = union - intersection
    OE_area = y_sim - intersection
    UE_area = y_obs - intersection

    IIEE = np.sum(IIEE_area == 1) * 625 / 1e6
    BACC = 1 - IIEE / (27207 * 625 / 1e6)

    OE = np.sum(OE_area == 1) * 625 / 1e6
    UE = np.sum(UE_area == 1) * 625 / 1e6

    return OE, UE, IIEE


def calculate_IIEE_BACC(y_prediction, y_true, start_date='2016-01'):
    y_prediction[y_prediction >= 0.15] = 1
    y_prediction[y_prediction < 0.15] = 0

    y_true[y_true >= 0.15] = 1
    y_true[y_true < 0.15] = 0

    # 保存每个月的误差总值
    IIEE_all = np.zeros(12)
    OE_all = np.zeros(12)
    UE_all = np.zeros(12)
    # 统计测试集中各个月份的分布
    month_distribution = np.zeros(12)
    # 开始的时间
    pd_start_date = pd.to_datetime(start_date)

    # 遍历每一个样本
    for i in range(y_true.shape[-1]):
        # 样本对应的起始时间，预测窗口为6个月，step为1个月
        sample_date = pd_start_date + pd.DateOffset(months=i)
        sample_month = sample_date.month
        OE, UE, IIEE = calculate_IIEE_single(y_true[:, :, i], y_prediction[:, :, i])

        IIEE_all[sample_month - 1] += IIEE
        OE_all[sample_month - 1] += OE
        UE_all[sample_month - 1] += UE

        # 统计月份
        month_distribution[sample_month - 1] += 1


    IIEE_all = IIEE_all / month_distribution
    OE_all = OE_all / month_distribution
    UE_all = UE_all / month_distribution

    # 持久化SIC loss
    return OE_all, UE_all, IIEE_all


def calculate_IIEE_BACC_persistence():
    agg_sic = np.load('../../data_preprocess/SIC/g2202_197901_202312.npy')
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0
    obs = agg_sic[:, :, 12 * (2016 - 1979):12 * (2023 - 1979)]

    IIEE_list = []
    OE_list = []
    UE_list = []

    for i in range(1, 7):
        prd = np.load('G2202_persistence_forcast_lead_month_' + str(i) + '.npy')[:, :, :-12]
        OE, UE, IIEE = calculate_IIEE_BACC(y_prediction=prd, y_true=obs, start_date='2016-01')
        IIEE_list.append(IIEE)
        OE_list.append(OE)
        UE_list.append(UE)


    IIEE_list = np.array(IIEE_list)
    OE_list = np.array(OE_list)
    UE_list = np.array(UE_list)


    print('IIEE:')
    print(np.mean(IIEE_list))
    print('OE:')
    print(np.mean(OE_list))
    print('UE:')
    print(np.mean(UE_list))

    # IceMamba_IIEE = np.load('../../SICMamba/result/IIEE_IceMamba-1.npy')


    plt.rcParams['axes.labelsize'] = 25  # 坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 25  # x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 25  # y轴刻度字体大小
    plt.rcParams['figure.titlesize'] = 25  # 图表标题字体大小
    plt.rcParams['font.size'] = 25  # colorbar字体大小

    plt.figure(figsize=(15, 10))
    plt.plot(np.mean(IIEE_list, axis=0), marker='x', label='Anomaly Persistence')
    # plt.plot(IceMamba_IIEE, marker='x', label='IceMamba-1')  # 请注意这里有一个小错误，应该是label而不是lable

    plt.grid()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
               labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.ylabel('IIEE ($10^6 km^2$)')
    plt.xlabel('Lead Time (Month)')
    plt.legend()
    plt.savefig('IIEE.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

    np.savez('persistence_IIEE',
             IIEE=IIEE_list,
             OE=OE_list,
             UE=UE_list)


def calculate_acc_signal(y_obs, y_sim):
    # 计算均值
    mask = np.load('../../data_preprocess/g2202_land.npy').astype(bool)
    y_obs = y_obs[mask, :]
    y_sim = y_sim[mask, :]
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


def calculate_acc_persistence_all():
    agg_sic = np.load('../../data_preprocess/SIC/g2202_197901_202312.npy')
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0
    agg_sic = agg_sic[:, :, :-12]
    # 2016/1 至 2022/12
    obs = agg_sic[:, :, 12 * (2016 - 1979):12 * (2023 - 1979)]
    obs = np.concatenate([obs, obs, obs, obs, obs, obs], axis=2)

    prd = np.concatenate([np.load('G2202_persistence_forcast_lead_month_1.npy')[:, :, :-12],
                          np.load('G2202_persistence_forcast_lead_month_2.npy')[:, :, :-12],
                          np.load('G2202_persistence_forcast_lead_month_3.npy')[:, :, :-12],
                          np.load('G2202_persistence_forcast_lead_month_4.npy')[:, :, :-12],
                          np.load('G2202_persistence_forcast_lead_month_5.npy')[:, :, :-12],
                          np.load('G2202_persistence_forcast_lead_month_6.npy')[:, :, :-12]], axis=2)
    print('ACC')
    print(calculate_acc_signal(obs, prd))


def calculate_acc(y_prediction, y_true, start_date='2016-01'):
    # 保存每个月的ACC总值
    acc_all = np.zeros(12)
    # 统计测试集中各个月份的分布
    month_distribution = np.zeros(12)
    # 存储每个月的真实值和预测值
    obs_by_month = {month: [] for month in range(1, 13)}

    # 开始的时间
    pd_start_date = pd.to_datetime(start_date)

    # 遍历每一个样本
    for i in range(y_true.shape[-1]):
        # 样本对应的起始时间
        sample_date = pd_start_date + pd.DateOffset(months=i)
        sample_month = sample_date.month

        # 提取真实值和预测值
        y_obs = y_true[:, :, i]
        y_prd = y_prediction[:, :, i]

        # 存储提取的值
        obs_by_month[sample_month].append((y_obs, y_prd))
        month_distribution[sample_month - 1] += 1

    # 计算ACC
    for month in range(1, 13):
        if month_distribution[month - 1] > 0:
            combined_obs = []
            combined_prd = []
            for obs, prd in obs_by_month[month]:
                combined_obs.append(obs)  # 扁平化并合并
                combined_prd.append(prd)

            if len(combined_obs) > 0 and len(combined_prd) > 0:
                np_combined_obs = np.transpose(np.array(combined_obs), (1, 2, 0))
                np_combined_prd = np.transpose(np.array(combined_prd), (1, 2, 0))
                acc = calculate_acc_signal(y_obs=np_combined_obs, y_sim=np_combined_prd)# 计算相关系数
                acc_all[month - 1] = acc

    return acc_all


def calculate_acc_persistence():
    agg_sic = np.load('../../data_preprocess/SIC/g2202_197901_202312.npy')
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0
    obs = agg_sic[:, :, 12 * (2016 - 1979):(2023 - 1979) * 12]

    acc_list = []

    for i in range(1, 7):
        prd = np.load('G2202_persistence_forcast_lead_month_' + str(i) + '.npy')[:, :, :-12]
        acc = calculate_acc(prd, obs)
        acc_list.append(acc)

    acc_list = np.array(acc_list)
    acc_list_month = np.mean(acc_list, axis=0)
    acc_list_lead = np.mean(acc_list, axis=1)

    np.save('G2202_persistence_ACC_lead_time', acc_list_lead)

    np.save('G2202_persistence_ACC_month', acc_list_month)

    np.save('G2202_persistence_ACC', acc_list)




if __name__ == '__main__':
    anomaly_persistence_forecast()
    calculate_mae_rmse_persistence()
    calculate_acc_persistence_all()
    calculate_IIEE_BACC_persistence()
    calculate_acc_persistence()


