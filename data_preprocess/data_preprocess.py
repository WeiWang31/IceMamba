import gc
import requests
import pandas as pd
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt


def transform_sic_nc_to_numpy():
    ds = xr.open_dataset('NSIDC_G2202/seaice_conc_monthly_nh_197811_202312_v04r00.nc')
    sic = ds['cdr_seaice_conc_monthly'].data
    # sic = np.transpose(sic, (1, 2, 0))
    print(sic[2:, :, :].shape)
    # preprocess the SIC data
    sic[sic > 1] = 0
    sic[np.isnan(sic)] = 0
    sic = np.transpose(sic, (1, 2, 0))

    if not os.path.exists('./SIC'):
        os.makedirs('./SIC')
    else:
        pass

    np.save('./SIC/g2202_197901_202312', sic[:, :, 2:])


def generate_land_mask():
    path = 'NSIDC_G2202/seaice_conc_monthly_nh_197811_202312_v04r00.nc'
    ds = xr.open_dataset(path)
    sic = ds['cdr_seaice_conc_monthly'].data
    mask = np.ones(shape=(448, 304))
    mask[sic[0] == 2.54] = 0
    print(mask.shape)
    np.save('g2202_land', mask)
    plt.imshow(mask)
    plt.show()


# 计算1979年至2020年（训练集的时间范围）的 9月的SIC std
# 仅计算9月的std
# 范围为 [1979/9, 2020/9]
def calculate_sic_std(agg_sic):
    agg_sic[np.isnan(agg_sic)] = 0
    agg_sic[agg_sic > 1] = 0
    # 范围为 [1979/9, 2020/9]
    std = np.std(agg_sic[:, :, 8:(2020 - 1979 + 1)*12:12], axis=(2))

    plt.imshow(std)
    plt.show()
    plt.close()
    std_mask = np.zeros(shape=std.shape)
    std_mask[std > 0.1] = 1
    plt.imshow(std_mask)
    plt.show()
    plt.close()
    np.save('std_mask', std_mask)


# 计算1979/01年至2023/12的最大SIE，用于计算BACC
def print_max_SIE_area_g2202():
    agg_sic = np.load('./SIC/g2202_197901_202312.npy')
    print(np.max(np.sum(agg_sic >= 0.15, axis=(0, 1)), axis=0))


def preprocess_SIC():
    transform_sic_nc_to_numpy()
    generate_land_mask()
    calculate_sic_std(agg_sic=np.load('./SIC/g2202_197901_202312.npy'))
    print_max_SIE_area_g2202()
    print('SIC preprocessing done!')


# --------------------ERA5 and ORAS5 data preprocess--------------------------
# 将.nc 文件转化为numpy数组
def ear5_oras5_to_numpy(ds_name='ERA5'):
    ds_path = ds_name + '_EASE/'

    ds_file_list = os.listdir(ds_path)
    ds_file_list.sort()
    if ds_name == 'ERA5':
        ds_keys = ['msl', 'ssrd', 'rsus', 't', 't2m', 'sst', 'u', 'u10', 'v10', '__xarray_dataarray_variable__',
                   '__xarray_dataarray_variable__']
    elif ds_name == 'ORAS5':
        ds_keys = ['__xarray_dataarray_variable__',
                   '__xarray_dataarray_variable__',
                   '__xarray_dataarray_variable__',
                   '__xarray_dataarray_variable__']

    if not os.path.exists(ds_name + '/abs'):
        os.makedirs(ds_name + '/abs')
    else:
        pass

    for i in range(len(ds_file_list)):
        ds_file = xr.open_dataset(ds_path + ds_file_list[i])
        ds_file_name = ds_file_list[i].split('.')[0]
        ds_file_name = ds_name + '/abs/' + ds_file_name.split('_')[0] + '_1979-01_2022-12'
        print(np.transpose(ds_file[ds_keys[i]].data, axes=(1, 2, 0)).shape)
        print(ds_file_name)
        np.save(ds_file_name, np.transpose(ds_file[ds_keys[i]].data, axes=(1, 2, 0)))

    print(ds_name + 'to numpy done!')


# anomaly the era5 or oras5 data
def ear5_oras5_to_anomaly(ds_name='ERA5'):
    if ds_name != 'ERA5' and ds_name != 'ORAS5':
        print('Please chose ERA5 or ORAS5!')
    else:
        if os.path.exists(ds_name + '/abs'):
            if not os.path.exists(ds_name + '/anomaly'):
                os.makedirs(ds_name + '/anomaly')
            path = ds_name + '/abs/'
            file_list = os.listdir(path)
            for file_name in file_list:
                save_file_name = ds_name + '/anomaly/' + file_name.split('.')[0] + '_anomaly'
                anomaly_data(np.load(path + file_name), save_file_name)
        else:
            print('出现错误，请检查 ./' + ds_name + '/abs 路径是否存在')


def anomaly_data(ds, name):
    # 计算1979年1月至2010年12月每个月的平均值, 对应训练集的数据

    ds_full = ds
    ds = ds[:, :, :384]
    monthly_means = []
    for month in range(12):
        # 选取每个月的数据
        month_data = ds[:, :, month::12]
        # 计算1979年1月至2010年12月的平均值
        mean = np.mean(month_data, axis=2)
        monthly_means.append(mean)

    # 将列表转换为numpy数组
    monthly_means = np.array(monthly_means)

    # 初始化一个与原始数组形状相同的数组来存储异常值
    anomalies = np.empty_like(ds_full)

    # 计算异常值
    for month in range(12):
        # 找出所有该月份的索引
        month_indices = np.arange(month, ds_full.shape[2], 12)
        # 计算每个月的异常值并存储
        anomalies[:, :, month_indices] = ds_full[:, :, month_indices] - monthly_means[month, :, :, None]

    np.save(name, anomalies)

# normalize the era5 or oras5 data
def normalize_data(path, ds_name='ERA5', is_abs=False):
    file_list = os.listdir(path)
    file_list.sort()

    for name in file_list:
        anomaly_data = np.load(path + name)
        mean = np.nanmean(anomaly_data[:, :, :384], dtype=np.float64)
        std = np.nanstd(anomaly_data[:, :, :384], dtype=np.float64)
        normalized_data = (anomaly_data - mean) / std
        if is_abs:
            if not os.path.exists(ds_name + '/normalized_abs'):
                os.makedirs(ds_name + '/normalized_abs')
                np.save(ds_name + '/normalized_abs/' + 'normalized_' + name.split('.')[0], normalized_data)
            else:
                np.save(ds_name + '/normalized_abs/' + 'normalized_' + name.split('.')[0], normalized_data)
        else:
            if not os.path.exists(ds_name + '/normalized'):
                os.makedirs(ds_name + '/normalized')
                np.save(ds_name + '/normalized/' + 'normalized_' + name.split('.')[0], normalized_data)
            else:
                np.save(ds_name + '/normalized/' + 'normalized_' + name.split('.')[0], normalized_data)


# 数据的处理流程：1.下载；2.regrid; 3.uas, vas 特别预处理； 4.取anomaly 5. 取normalize
def process_era5_oras5():
    for ds_name in ['ERA5', 'ORAS5']:
        ear5_oras5_to_numpy(ds_name)
        ear5_oras5_to_anomaly(ds_name)
        normalize_data(path=ds_name + '/abs/', ds_name=ds_name, is_abs=True)
        normalize_data(path=ds_name + '/anomaly/', ds_name=ds_name, is_abs=False)
        print(ds_name + 'preprocess done !')


if __name__ == '__main__':
    preprocess_SIC()
    process_era5_oras5()


