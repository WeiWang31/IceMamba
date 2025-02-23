import argparse

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from IceMamba import VSSM
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import json


class MyDataset(Dataset):
    def __init__(self, var_dict, mask, time_dict, num_forecast):
        super(MyDataset, self).__init__()
        self.var_dict = var_dict
        self.mask = mask
        self.time_dict = time_dict
        self.in_chans = int(self.time_dict['sic'])
        self.num_forecast = num_forecast

    def __getitem__(self, index):
        for var_name in self.var_dict.keys():
            if var_name == 'sic':
                item = self.var_dict[var_name][index:index + self.in_chans]
            else:
                item = torch.cat((item, self.var_dict[var_name][index + self.in_chans - self.time_dict[var_name]:index + self.in_chans]), dim=0)

        label = self.var_dict['sic'][index + self.in_chans:index + self.num_forecast + self.in_chans]
        item_mask = self.mask
        return item, label, item_mask

    def __len__(self):
        return self.var_dict['sic'].shape[0] - (self.in_chans + self.num_forecast) + 1


# 注意，这里的year range范围是[start_year/1, end_year/12]
def data_transform(NSIDC_path, ERA5_path, ORAS5_path, mask_path,
                   time_dict, preprocess_dict,
                   num_forecast,
                   bs, shuffle=True, year_range=''):
    if year_range != '':
        start_year = int(year_range.split('-')[0])
        end_year = int(year_range.split('-')[1])
        start_index = (start_year - 1979) * 12
        end_index = (end_year - 1979 + 1) * 12

        var_dict = {}

        # 转化为torch 张量顺序 (B, H, W, C) -> (B, C, H, W)
        var_list = time_dict.keys()

        for var_name in var_list:
            preprocess = preprocess_dict[var_name]

            if var_name == 'sic':
                var_path = NSIDC_path
            elif var_name in ['tas', 'ta500', 'tos', 'rsds', 'rsus', 'psl', 'zg500', 'zg250', 'ua10', 'uas', 'vas']:
                var_path = ERA5_path
                if preprocess == 'abs':
                    var_path = var_path + 'normalized_abs/normalized_' + var_name + '_1979-01_2022-12.npy'
                elif preprocess == 'anomaly':
                    var_path = var_path + 'normalized/normalized_' + var_name + '_1979-01_2022-12_anomaly.npy'
            elif var_name in ['ohc300', 'ohc700', 'mld001', 'mld003']:
                var_path = ORAS5_path
                if preprocess == 'abs':
                    var_path = var_path + 'normalized_abs/normalized_' + var_name + '_1979-01_2022-12.npy'
                elif preprocess == 'anomaly':
                    var_path = var_path + 'normalized/normalized_' + var_name + '_1979-01_2022-12_anomaly.npy'

            var = np.load(var_path)
            var = np.transpose(var, [2, 0, 1])
            var = var[start_index:end_index, :, :]
            var = torch.tensor(var, dtype=torch.float)
            var_dict[var_name] = var

        sample_mask = np.load(mask_path)
        sample_mask = np.reshape(sample_mask, (1, sample_mask.shape[0], sample_mask.shape[1]))
        sample_mask = np.repeat(sample_mask, num_forecast, axis=0)
        sample_mask = torch.from_numpy(sample_mask)
        mask = sample_mask.ge(1)
        train_data = MyDataset(var_dict, mask, time_dict, num_forecast)
        train_data = DataLoader(train_data, batch_size=bs, shuffle=shuffle)

        return train_data

# 用于 target year的9月预测
def data_transform_sipn(NSIDC_path, ERA5_path, ORAS5_path, mask_path,
                   time_dict, preprocess_dict,
                   num_forecast,
                   bs, shuffle=True, target_year=2001):
    if 2000 < target_year < 2021:
        start_index = (target_year - 1979) * 12 - 3 - num_forecast
        end_index = (target_year - 1979) * 12 + 9 + num_forecast - 1
        var_dict = {}

        # 转化为torch 张量顺序 (B, H, W, C) -> (B, C, H, W)
        var_list = time_dict.keys()

        for var_name in var_list:
            preprocess = preprocess_dict[var_name]
            if var_name == 'sic':
                var_path = NSIDC_path
            elif var_name in ['tas', 'ta500', 'tos', 'rsds', 'rsus', 'psl', 'zg500', 'zg250', 'ua10', 'uas', 'vas']:
                var_path = ERA5_path
                if preprocess == 'abs':
                    var_path = var_path + 'normalized_abs/normalized_' + var_name + '_1979-01_2022-12.npy'
                elif preprocess == 'anomaly':
                    var_path = var_path + 'normalized/normalized_' + var_name + '_1979-01_2022-12_anomaly.npy'
            elif var_name in ['ohc300', 'ohc700', 'mld001']:
                var_path = ORAS5_path
                if preprocess == 'abs':
                    var_path = var_path + 'normalized_abs/normalized_' + var_name + '_1979-01_2022-12.npy'
                elif preprocess == 'anomaly':
                    var_path = var_path + 'normalized/normalized_' + var_name + '_1979-01_2022-12_anomaly.npy'

            var = np.load(var_path)
            var = np.transpose(var, [2, 0, 1])
            var = var[start_index:end_index, :, :]
            var = torch.tensor(var, dtype=torch.float)
            var_dict[var_name] = var

        sample_mask = np.load(mask_path)
        sample_mask = np.reshape(sample_mask, (1, sample_mask.shape[0], sample_mask.shape[1]))
        sample_mask = np.repeat(sample_mask, num_forecast, axis=0)
        sample_mask = torch.from_numpy(sample_mask)
        mask = sample_mask.ge(1)
        train_data = MyDataset(var_dict, mask, time_dict, num_forecast)
        train_data = DataLoader(train_data, batch_size=bs, shuffle=shuffle, num_workers=20)

        return train_data


class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def calculate_IIEE(self, y_sim, y_obs):
        y_obs_SIE = torch.zeros(y_obs.shape)
        y_sim_SIE = torch.zeros(y_sim.shape)

        y_obs_SIE[y_obs >= 0.15] = 1
        y_obs_SIE[y_obs < 0.15] = 0
        y_sim_SIE[y_sim >= 0.15] = 1
        y_sim_SIE[y_sim < 0.15] = 0

        union = y_sim_SIE + y_obs_SIE
        union[union == 2] = 1
        intersection = y_obs_SIE * y_sim_SIE
        IIEE_grid = union - intersection

        # 最后的结果要除以 Batch * Channel
        IIEE = torch.sum(IIEE_grid == 1, dim=(2, 3)) * 625 / 1e6
        IIEE = torch.mean(IIEE)

        BACC = 1 - IIEE / (27207 * 625 / 1e6)

        return IIEE, BACC

    def forward(self, preds, target, mask):
        masked_preds = torch.masked_select(preds, mask)
        masked_target = torch.masked_select(target, mask)

        masked_mae = self.mae(masked_preds, masked_target)
        masked_mse = self.mse(masked_preds, masked_target)
        masked_rmse = torch.sqrt(masked_mse)

        zero_mask_preds = preds.clone()
        zero_mask_target = target.clone()

        zero_mask_preds[mask == False] = 0
        zero_mask_target[mask == False] = 0

        loss = masked_mae

        IIEE, BACC = self.calculate_IIEE(zero_mask_preds, zero_mask_target)

        return masked_mae, masked_rmse, IIEE, BACC, loss


class light_model(pl.LightningModule):
    def __init__(self, in_chans, num_forecast, patchembed_version='v1', downsample_version='v1', batch_size=2, lr=1e-4, ssm_drop_rate=0.0,
                 mlp_drop_rate=0.0, loss=MaskedLoss()):
        super().__init__()
        self.patchembed_version = patchembed_version
        self.downsample_version = downsample_version
        self.batch_size = batch_size
        self.loss = loss
        self.lr = lr
        self.accuracy = torchmetrics.MeanAbsoluteError()
        self.model = VSSM(in_chans=in_chans,
                          num_forecast=num_forecast,
                          patchembed_version=self.patchembed_version,
                          downsample_version=self.downsample_version,
                          ssm_drop_rate=ssm_drop_rate,
                          mlp_drop_rate=mlp_drop_rate)
        self.mse = torchmetrics.regression.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mask_loss = MaskedLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        (mae, rmse,
         IIEE, BACC,
         training_loss
         ) = self.loss(logits, y, mask)

        tensorboard = self.logger.experiment
        tensorboard.add_scalars('loss', {'train': training_loss}, self.global_step)
        tensorboard.add_scalars('MAE', {'train': mae}, self.global_step)
        tensorboard.add_scalars('RMSE', {'train': rmse}, self.global_step)
        tensorboard.add_scalars('IIEE', {'train': IIEE}, self.global_step)
        tensorboard.add_scalars('BACC', {'train': BACC}, self.global_step)

        self.log('training_loss', training_loss, prog_bar=True)
        self.log('training_MAE', mae, prog_bar=True)
        self.log('training_RMSE', rmse, prog_bar=True)
        self.log('training_IIEE', IIEE, prog_bar=True)
        self.log('training_BACC', BACC, prog_bar=True)

        return training_loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        (mae, rmse,
         IIEE, BACC,
         val_loss
         ) = self.loss(logits, y, mask)

        tensorboard = self.logger.experiment
        tensorboard.add_scalars('loss', {'val': val_loss}, self.global_step)
        tensorboard.add_scalars('MAE', {'val': mae}, self.global_step)
        tensorboard.add_scalars('RMSE', {'val': rmse}, self.global_step)
        tensorboard.add_scalars('IIEE', {'val': IIEE}, self.global_step)
        tensorboard.add_scalars('BACC', {'val': BACC}, self.global_step)

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_MAE', mae, prog_bar=True)
        self.log('val_RMSE', rmse, prog_bar=True)
        self.log('val_IIEE', IIEE, prog_bar=True)
        self.log('val_BACC', BACC, prog_bar=True)

        return {"val_loss": val_loss,
                "val_MAE": mae,
                "val_RMSE": rmse,
                "val_IIEE": IIEE,
                "val_BACC": BACC
                }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_MAE = torch.stack([x['val_MAE'] for x in outputs]).mean()
        avg_RMSE = torch.stack([x['val_RMSE'] for x in outputs]).mean()
        avg_IIEE = torch.stack([x['val_IIEE'] for x in outputs]).mean()
        avg_BACC = torch.stack([x['val_BACC'] for x in outputs]).mean()

        return {"val_loss": avg_loss,
                "val_MAE": avg_MAE,
                "val_RMSE": avg_RMSE,
                "val_IIEE": avg_IIEE,
                "val_BACC": avg_BACC
                }

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)
        logits[mask == 0] = 0
        logits[logits < 0] = 0
        (mae, rmse,
         IIEE, BACC,
         test_loss
         ) = self.loss(logits, y, mask)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_MAE', mae, prog_bar=True)
        self.log('test_RMSE', rmse, prog_bar=True)
        self.log('test_IIEE', IIEE, prog_bar=True)
        self.log('test_BACC', BACC, prog_bar=True)

        return {"test_loss": test_loss,
                "test_MAE": mae,
                "test_RMSE": rmse,
                "test_IIEE": IIEE,
                "test_BACC": BACC
                }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _, mask = batch
        x_hat = self(x)
        x_hat[mask == 0] = 0
        x_hat[x_hat < 0] = 0
        return x_hat

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
        return {'optimizer': optim, 'lr_scheduler': scheduler}


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_range', default='1979-2010')
    parser.add_argument('--val_range', default='2011-2014')
    parser.add_argument('--num_forecast', default=4)
    parser.add_argument('--time_dict_path', default='../../train_json/time_test.json')
    parser.add_argument('--preprocess_dict_path', default='../../train_json/preprocess_test.json')
    parser.add_argument('--train_type', default=1)
    
    args = parser.parse_args()
    train_range = args.train_range
    val_range = args.val_range
    num_forecast = int(args.num_forecast)
    train_type = args.train_type
    print('train_range: ' + train_range)
    print('val_range: ' + val_range)
    
    with open(args.time_dict_path, 'r') as file:
        time_dict = json.load(file)
    
    with open(args.preprocess_dict_path, 'r') as file:
        preprocess_dict = json.load(file)

    
    NSIDC_path = '../../data_preprocess/SIC/g2202_197901_202312.npy'
    ERA5_path = '../../data_preprocess/ERA5/'
    ORAS5_path = '../../data_preprocess/ORAS5/'
    mask_path = '../../data_preprocess/g2202_land.npy'

    seed = 42
    seed_everything(seed, workers=True)
    batch_size = 1
    in_chans = sum(time_dict.values())

    model = light_model(in_chans=in_chans, num_forecast=num_forecast, lr=1e-3, ssm_drop_rate=0.0, mlp_drop_rate=0.0)
    check_point = ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="best_sm_{epoch:02d}", mode="min")
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10, verbose=False, mode="min")

    logger = TensorBoardLogger('tb_logs_icemamba_' + str(train_type), name='icemamba')

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        callbacks=[early_stop, check_point],
        logger=logger
    )
    # 备忘，0最佳，50, ssm_drop_rate=0.0, mlp_drop_rate=0.0
    # 备忘，1，60, ssm_drop_rate=0.0, mlp_drop_rate=0.0，降低

    trainer.fit(model,
                train_dataloaders=data_transform(NSIDC_path,
                                                 ERA5_path,
                                                 ORAS5_path,
                                                 mask_path,
                                                 time_dict,
                                                 preprocess_dict,
                                                 num_forecast=num_forecast,
                                                 bs=batch_size,
                                                 shuffle=True,
                                                 year_range=train_range),

                val_dataloaders=data_transform(NSIDC_path,
                                               ERA5_path,
                                               ORAS5_path,
                                               mask_path,
                                               time_dict,
                                               preprocess_dict,
                                               num_forecast=num_forecast,
                                               bs=batch_size,
                                               shuffle=True,
                                               year_range=val_range),
                )

    print(check_point.best_model_path)
    




if __name__ == '__main__':
    train()
    
