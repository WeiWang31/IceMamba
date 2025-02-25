import argparse
import json
from IceMamba_train import light_model
from IceMamba_train import data_transform_sipn
from pytorch_lightning import Trainer
import numpy as np
import os

def get_sep_y(sic_path, num_forecast, target_year=2021):
    index = (target_year - 1979) * 12 + 8
    sic = np.load(sic_path)
    test_y = sic[:, :, index]
    return test_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_year', default='2001')
    parser.add_argument('--num_forecast', default=4)
    parser.add_argument('--time_dict_path', default='./train_json/time_dict.json')
    parser.add_argument('--preprocess_dict_path', default='./train_json/preprocess_dict.json')
    parser.add_argument('--ckpt_path', default='')
    

    args = parser.parse_args()
    target_year = int(args.target_year)
    ckpt_path = args.ckpt_path
    num_forecast = int(args.num_forecast)
    
    print('target_year: ' + str(target_year))

    
    with open(args.time_dict_path, 'r') as file:
        time_dict = json.load(file)
    
    with open(args.preprocess_dict_path, 'r') as file:
        preprocess_dict = json.load(file)
    
    in_chans = sum(time_dict.values())

   
    NSIDC_path = '../data_preprocess/SIC/g2202_197901_202312.npy'
    ERA5_path = '../data_preprocess/ERA5/'
    ORAS5_path = '../data_preprocess/ORAS5/'
    mask_path = '../data_preprocess/g2202_land.npy'
    
    model = light_model.load_from_checkpoint(ckpt_path, in_chans=in_chans, num_forecast=num_forecast, lr=1e-3, ssm_drop_rate=0.0, mlp_drop_rate=0.0)
    model.eval()
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
    )
    
    preds = trainer.predict(model, data_transform_sipn(NSIDC_path,
                                                 ERA5_path,
                                                 ORAS5_path,
                                                 mask_path,
                                                 time_dict,
                                                 preprocess_dict,
                                                 num_forecast=num_forecast,
                                                 bs=1,
                                                 shuffle=False,
                                                 target_year=target_year))

    preds_list = []
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = pred.view(448, 304, num_forecast)
        np_pred = pred.numpy()
        preds_list.append(np_pred)
    np_preds = np.array(preds_list)
    print(np_preds.shape)
    if os.path.exists('./SIPN_evaluation_result'):
        pass
    else:
        os.mkdir('./SIPN_evaluation_result')
    y = get_sep_y(NSIDC_path, num_forecast, target_year)
    
    np.savez('./SIPN_evaluation_result/SIPN_prd_' + str(target_year),
             prd=np_preds,
             Y=y)