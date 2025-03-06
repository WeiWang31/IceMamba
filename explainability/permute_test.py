import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from Icemamba import VSSM
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning import Trainer
import os


class MyDataset(Dataset):
    def __init__(self, feature, target, mask):
        super(MyDataset, self).__init__()
        self.feature = feature
        self.target = target
        self.mask = mask

    def __getitem__(self, index):
        item = self.feature[index]
        label = self.target[index]
        item_mask = self.mask[index]
        return item, label, item_mask

    def __len__(self):
        return len(self.feature)


# 注意，这里的year range范围是[start_year/1, end_year/12]
def data_transform(data_set, mask_path, bs, shuffle=True, permute_channel=-1, seed=-1):
    training_set = np.load(data_set)
    # 转化为torch 张量顺序 (B, H, W, C) -> (B, C, H, W)
    x = torch.tensor(np.transpose(training_set['X'], (0, 3, 1, 2)), dtype=torch.float)
    y = torch.tensor(np.transpose(training_set['Y'], (0, 3, 1, 2)), dtype=torch.float)

    if permute_channel != -1:
        if seed != -1:
            torch.manual_seed(seed)
        channel_data = x[:, permute_channel, :, :]
        shuffled_data = channel_data[torch.randperm(channel_data.size(0))]

        x[:, permute_channel, :, :] = shuffled_data
    else:
        pass

    num_sample = x.shape[0]
    sample_mask = np.load(mask_path)
    sample_mask = np.reshape(sample_mask, (1, 1, sample_mask.shape[0], sample_mask.shape[1]))
    sample_mask = torch.from_numpy(sample_mask)
    mask = sample_mask.ge(1)
    mask = mask.repeat(1, 6, 1, 1)
    mask = mask.repeat(num_sample, 1, 1, 1)

    train_data = MyDataset(x, y, mask)
    train_data = DataLoader(train_data, batch_size=bs, shuffle=shuffle)

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
    def __init__(self, in_chans, num_forecast, patchembed_version='v1', downsample_version='v1', batch_size=2, lr=1e-4,
                 ssm_drop_rate=0.0,
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


def _feature_permute(trainer, model, test_set_path, name, permute_channel, seed, is_ua10=False):
    # test = trainer.test(model,
    #                     data_transform(
    #                         test_set_path,
    #                         mask_path='../data_preprocess/g2202_land.npy',
    #                         bs=1,
    #                         shuffle=False,
    #                         permute_channel=permute_channel,
    #                         seed=seed,
    #                     ))

    preds = trainer.predict(model,
                            data_transform(
                                test_set_path,
                                mask_path='../data_preprocess/g2202_land.npy',
                                bs=1,
                                shuffle=False, permute_channel=permute_channel,
                                seed=seed,
                            ))

    preds_list = []
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.view(448, 304, 6)
        np_pred = pred.numpy()
        preds_list.append(np_pred)
    np_preds = np.array(preds_list)
    print(np_preds.shape)

    result_path = ''
    if is_ua10:
        result_path = './result_ua10'
    else:
        result_path = './result'

    if seed == -1:
        path = result_path + '/org/'
    else:
        path = result_path + '/seed_' + str(seed) + '/'

    if os.path.exists(path):
        np.save(path + name, np_preds)
    else:
        os.makedirs(path)
        np.save(path + name, np_preds)


def feature_permute(seed, ckpt_path, test_set_path, in_chans=50, is_ua10=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_forecast', default=6)
    args = parser.parse_args()
    num_forecast = int(args.num_forecast)
    model = light_model.load_from_checkpoint(ckpt_path, in_chans=in_chans, num_forecast=num_forecast, lr=1e-3,
                                             ssm_drop_rate=0.0, mlp_drop_rate=0.0)
    model.eval()

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
    )
    # 50个变量
    for i in range(50):
        if seed == -1:
            name = 'prd_org'
            _feature_permute(trainer, model, test_set_path, name, -1, seed=seed, is_ua10=is_ua10)
            break
        else:
            name = 'prd_' + str(i) + '_seed_' + str(seed)
            _feature_permute(trainer, model, test_set_path, name, i, seed=seed, is_ua10=is_ua10)


if __name__ == '__main__':
    # 可解释性分析
    feature_permute(seed=-1,
                    ckpt_path='../Model/IceMamba/ckpt/best_sm_icemamba-6.ckpt',
                    test_set_path='./test_set_era5_combine_icemamba-6.npz',
                    in_chans=50,
                    is_ua10=False)

    for i in range(23, 33):
        feature_permute(seed=i,
                        ckpt_path='../Model/IceMamba/ckpt/best_sm_icemamba-6.ckpt',
                        test_set_path='./test_set_era5_combine_icemamba-6.npz',
                        in_chans=50,
                        is_ua10=False)

    # ua10 去趋势后的可解释性分析
    feature_permute(seed=-1,
                    ckpt_path='../Model/IceMamba/ckpt/best_sm_icemamba-6-ua10.ckpt',
                    test_set_path='./test_set_era5_combine_icemamba-6-ua10.npz',
                    in_chans=50,
                    is_ua10=True)

    for i in range(23, 33):
        feature_permute(seed=i,
                        ckpt_path='../Model/IceMamba/ckpt/best_sm_icemamba-6-ua10.ckpt',
                        test_set_path='./test_set_era5_combine_icemamba-6-ua10.npz',
                        in_chans=50,
                        is_ua10=True)

