import gc
import requests
import pandas as pd
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt


def _generate_combine_dataset(agg_sic, era5_dict={}, name='era5_combine'):
    test_set_X = []
    test_set_Y = []

    era5_keys = era5_dict.keys()

    for i in range(540):
        print(i)
        if i <= (2010 - 1979) * 12 - 6:
            pass

        elif (2011 - 1979) * 12 <= i <= (2014 - 1979) * 12 - 6:
            pass

        elif (2015 - 1979) * 12 <= i <= (2022 - 1979) * 12 - 6:
            test_X = []
            test_X.append(agg_sic[:, :, i:i + 12])
            for key in era5_keys:
                test_X.append(np.load(key)[:, :, i + (12 - int(era5_dict[key])):i + 12])
            test_X = np.concatenate(test_X, axis=2)
            test_set_X.append(test_X)
            test_set_Y.append(agg_sic[:, :, i + 12:i + 18])


    # 只保存测试集
    np.savez('./test_set' + name,
             X=test_set_X,
             Y=test_set_Y)


def generate_combine_dataset_ua10(agg_sic_path='../data_preprocess/SIC/g2202_197901_202312.npy',
                             era5_path='../data_preprocess/ERA5/normalized/',
                             oras5_path='../data_preprocess/ORAS5/normalized/',
                             lead_time=3,):
    agg_sic = np.load(agg_sic_path)

    reanalysis_variables_dict = {}
    reanalysis_variables_name_list = ['tas', 'ta500', 'tos', 'ohc700', 'mld001', 'mld003', 'rsds', 'rsus', 'psl', 'zg500', 'zg250', 'ua10', 'uas', 'vas']

    for file_name in reanalysis_variables_name_list:
        if file_name == 'uas' or file_name == 'vas':
            reanalysis_variables_dict[era5_path[:-1] + '_abs/normalized_' + file_name + '_1979-01_2022-12.npy'] = 1
        elif file_name in ['ohc700', 'mld001', 'mld003']:
            reanalysis_variables_dict[oras5_path + 'normalized_' + file_name + '_1979-01_2022-12_anomaly.npy'] = lead_time
        else:
            reanalysis_variables_dict[era5_path + 'normalized_' + file_name + '_1979-01_2022-12_anomaly.npy'] = lead_time

    name = '_era5_combine_icemamba-6-ua10'

    _generate_combine_dataset(agg_sic=agg_sic, era5_dict=reanalysis_variables_dict, name=name)

def generate_combine_dataset(agg_sic_path='../data_preprocess/SIC/g2202_197901_202312.npy',
                             era5_path='../data_preprocess/ERA5/normalized/',
                             oras5_path='../data_preprocess/ORAS5/normalized/',
                             lead_time=3,):
    agg_sic = np.load(agg_sic_path)

    reanalysis_variables_dict = {}
    reanalysis_variables_name_list = ['tas', 'ta500', 'tos', 'ohc700', 'mld001', 'mld003', 'rsds', 'rsus', 'psl', 'zg500', 'zg250', 'ua10', 'uas', 'vas']

    for file_name in reanalysis_variables_name_list:
        if file_name == 'uas' or file_name == 'vas':
            reanalysis_variables_dict[era5_path[:-1] + '_abs/normalized_' + file_name + '_1979-01_2022-12.npy'] = 1
        elif file_name == 'ua10':
            reanalysis_variables_dict[era5_path[:-1] + '_abs/normalized_' + file_name + '_1979-01_2022-12.npy'] = 3
        elif file_name in ['ohc700', 'mld001', 'mld003']:
            reanalysis_variables_dict[oras5_path + 'normalized_' + file_name + '_1979-01_2022-12_anomaly.npy'] = lead_time
        else:
            reanalysis_variables_dict[era5_path + 'normalized_' + file_name + '_1979-01_2022-12_anomaly.npy'] = lead_time

    name = '_era5_combine_icemamba-6'

    _generate_combine_dataset(agg_sic=agg_sic, era5_dict=reanalysis_variables_dict, name=name)


if __name__ == '__main__':
    generate_combine_dataset()
    generate_combine_dataset_ua10()
