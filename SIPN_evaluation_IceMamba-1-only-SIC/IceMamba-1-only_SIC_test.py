import argparse
import gc
import os

from torch.utils.data import DataLoader, Dataset
from IceMamba_train import light_model
from pytorch_lightning import Trainer
import numpy as np
import torch


# 返回lead time 1 - 4 的训练数据
def get_sep_data(NSIDC_path, mask_path, test_year=2001):
    # 注意右侧为开区间
    start_index = (test_year - 1979) * 12 - 7
    end_index = (test_year - 1979) * 12 - 3
    sic = np.load(NSIDC_path)
    sic = np.transpose(sic, [2, 0, 1])
    x = np.zeros(shape=(1, 12, 448, 304))

    for i in range(start_index, end_index):
        x = np.concatenate((x, np.expand_dims(sic[i:i + 12, :, :], axis=0)), axis=0)
    x = x[1:, :, :, :]

    # 转化为torch 张量顺序 (B, H, W, C) -> (B, C, H, W)
    x = torch.tensor(x, dtype=torch.float)
    # 该年对应的九月对应的SIC
    y = sic[(test_year - 1979) * 12 + 8, :, :]

    # 对应y的mask
    sample_mask = np.load(mask_path)
    sample_mask = np.reshape(sample_mask, (1, 1, sample_mask.shape[0], sample_mask.shape[1]))
    sample_mask = torch.from_numpy(sample_mask)
    mask = sample_mask.ge(1)
    mask = mask.repeat(1, 1, 1, 1)

    return x, y, mask


def predict_september(NSIDC_path, test_year=2001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = light_model.load_from_checkpoint('./IceMamba_SIPN_ckpt/best_sm_' + str(test_year) + '.ckpt',
                                             in_chans=12,
                                             num_forecast=1,
                                             lr=1e-3,
                                             ssm_drop_rate=0.0,
                                             mlp_drop_rate=0.0)

    model.to(device)
    model.eval()

    x, y, mask = get_sep_data(NSIDC_path=NSIDC_path, mask_path='../data_preprocess/g2202_land.npy', test_year=test_year)
    x = x.to(device)
    mask = mask.to(device)

    sep_prd = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            # lead time = 4
            if i == 0:
                x_item = x[0:1]
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 1
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 2
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 3
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                sep_prd.append(y_hat)

            # lead time = 3
            elif i == 1:
                x_item = x[1:2]
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 1
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 2
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                sep_prd.append(y_hat)

            # lead time = 2
            elif i == 2:
                x_item = x[2:3]
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                # 将预测值添加至末尾进行循环预测 1
                x_item = torch.cat((x_item[:, 1:, :, :], y_hat), dim=1)
                y_hat = model(x_item)
                # 去除陆地部分
                y_hat[mask == False] = 0
                sep_prd.append(y_hat)

            # lead time = 1
            elif i == 3:
                y_hat = model(x[3:4])
                # 去除陆地部分
                y_hat[mask == False] = 0
                sep_prd.append(y_hat)

    spe_prd = torch.cat(sep_prd, dim=1)
    spe_prd = spe_prd.permute(0, 2, 3, 1)
    spe_prd = spe_prd.view(448, 304, 4)
    spe_prd = spe_prd.cpu()
    spe_prd = spe_prd.numpy()
    if os.path.exists('./SIPN_evaluation_result'):
        pass
    else:
        os.mkdir('./SIPN_evaluation_result')

    np.savez('./SIPN_evaluation_result/SIPN_prd_' + str(test_year),
             prd=spe_prd,
             y=y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_year', default='2016')

    args = parser.parse_args()
    test_year = args.test_year

    for i in range(2001, 2021):
        print(i)
        predict_september(NSIDC_path='../data_preprocess/SIC/g2202_197901_202312.npy', test_year=i)

