import argparse
import json

from IceMamba_train import light_model
from IceMamba_train import data_transform
from pytorch_lightning import Trainer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


def get_test_y(sic_path, num_forecast, year_range='2015-2022'):
    start_year = int(year_range.split('-')[0])
    end_year = int(year_range.split('-')[1])
    start_index = (start_year - 1979) * 12
    end_index = (end_year - 1979 + 1) * 12
    sic = np.load(sic_path)
    sic = sic[:, :, start_index:end_index]
    test_y = []
    
    for i in range(sic.shape[-1] - 12 - num_forecast + 1):
        test_y.append(sic[:, :, i + 12:i + 12 + num_forecast])
    test_y = np.array(test_y)
    print(test_y.shape)
    return test_y

        
def calculate_SIC_MAE_MSE_order_by_month_lead_time(y_prediction, y_true, land_mask, file_name, start_date='2016-01'):
    lead_time = y_true.shape[-1]

    # 保存每个月的误差总值
    MAE_all = np.zeros((lead_time, 12))
    MSE_all = np.zeros((lead_time, 12))
    # 统计测试集中各个月份的分布
    month_distribution = np.zeros((lead_time, 12))
    # 开始的时间
    pd_start_date = pd.to_datetime(start_date)

    # 遍历每一个样本
    for i in range(y_true.shape[0]):
        # 样本对应的起始时间，预测窗口为6个月，step为1个月
        sample_start_date = pd_start_date + pd.DateOffset(months=i)
        # 计算预测窗口的时间
        # 计算第一个lead time
        for j in range(y_true.shape[-1]):
            sample_date = sample_start_date + pd.DateOffset(months=j)
            sample_month = sample_date.month
            y_obs = y_true[i, :, :, j]
            mask_obs = y_obs[land_mask.astype(bool)]
            y_prd = y_prediction[i, :, :, j]
            mask_prd = y_prd[land_mask.astype(bool)]
            MAE = mean_absolute_error(mask_obs, mask_prd)
            MSE = mean_squared_error(mask_obs, mask_prd)
            MAE_all[j, sample_month - 1] += MAE
            MSE_all[j, sample_month - 1] += MSE
            # 统计月份
            month_distribution[j, sample_month - 1] += 1

    MAE_all = MAE_all / month_distribution
    MSE_all = MSE_all / month_distribution
    
    print('MAE month (1-12):')
    print(np.mean(MAE_all, axis=0))
    print('MAE lead (1-6):')
    print(np.mean(MAE_all, axis=1))

    # 持久化SIC loss
    np.savez(file_name,
             MAE=MAE_all,
             MSE=MSE_all)

def calculate_acc(y_obs, y_sim, land_mask):
    # 计算均值
    #
    y_obs = y_obs[:, land_mask.astype(bool), :]
    y_sim = y_sim[:, land_mask.astype(bool), :]
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
    print('acc:' + str(acc))
    return acc
    
    

if __name__ == "__main__":
    print('--------------------------------------------------------------------------------------')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_range', default='2015-2022')
    parser.add_argument('--num_forecast', default=4)
    parser.add_argument('--time_dict_path', default='../../train_json/time_dict2.json')
    parser.add_argument('--preprocess_dict_path', default='../../train_json/preprocess_dict2.json')
    parser.add_argument('--ckpt_path', default='./ckpt/best_sm_icemamba-4.ckpt')
    parser.add_argument('--result_save_name', default='result_IceMamba-4')
    

    args = parser.parse_args()
    test_range = args.test_range
    num_forecast = int(args.num_forecast)
    ckpt_path = args.ckpt_path
    result_save_name = args.result_save_name
    
    print('test_range: ' + test_range)

    
    with open(args.time_dict_path, 'r') as file:
        time_dict = json.load(file)
    
    with open(args.preprocess_dict_path, 'r') as file:
        preprocess_dict = json.load(file)
    
    in_chans = sum(time_dict.values())

   
    NSIDC_path = '../../data_preprocess/SIC/g2202_197901_202312.npy'
    ERA5_path = '../../data_preprocess/ERA5/'
    ORAS5_path = '../../data_preprocess/ORAS5/'
    mask_path = '../../data_preprocess/g2202_land.npy'
    
    model = light_model.load_from_checkpoint(ckpt_path, in_chans=in_chans, num_forecast=num_forecast, lr=1e-3, ssm_drop_rate=0.0, mlp_drop_rate=0.0)
    model.eval()
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
    )
    test = trainer.test(model, data_transform(NSIDC_path,
                                                 ERA5_path,
                                                 ORAS5_path,
                                                 mask_path,
                                                 time_dict,
                                                 preprocess_dict,
                                                 num_forecast=num_forecast,
                                                 bs=1,
                                                 shuffle=False,
                                                 year_range=test_range))

    preds = trainer.predict(model, data_transform(NSIDC_path,
                                                 ERA5_path,
                                                 ORAS5_path,
                                                 mask_path,
                                                 time_dict,
                                                 preprocess_dict,
                                                 num_forecast=num_forecast,
                                                 bs=1,
                                                 shuffle=False,
                                                 year_range=test_range))

    preds_list = []
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.view(448, 304, num_forecast)
        np_pred = pred.numpy()
        preds_list.append(np_pred)
    np_preds = np.array(preds_list)
    print(np_preds.shape)
    
    land_mask = np.load('../../data_preprocess/g2202_land.npy')
    y_true = get_test_y(NSIDC_path, num_forecast)

    np.savez(result_save_name,
             prd=np_preds,
             Y=y_true,
             )
    calculate_SIC_MAE_MSE_order_by_month_lead_time(np_preds, y_true, land_mask, file_name='mse_mae_all_type_9-6')
    calculate_acc(np_preds, y_true, land_mask)

    print('--------------------------------------------------------------------------------------')

