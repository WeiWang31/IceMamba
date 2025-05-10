<div align="center">
<h1>Seasonal forecasting of Pan-Arctic sea ice with state space model </h1>
<h3>IceMamba: Arctic Sea Ice Forecasting Framework</h3>

[Wei Wang](https://github.com/WeiWang31)<sup>1</sup>,[WeiDong Yang]<sup>1</sup>,[Lei Wang]<sup>1</sup>, [GuiHua Wang]<sup>1</sup>, [Lei RuiBo]<sup>3</sup>, 

<sup>1</sup>  School of Computer Science, Fudan University, Shanghai, China, <sup>2</sup>  Department of Atmospheric and Oceanic Sciences & Institute of Atmospheric Sciences, Fudan
University, Shanghai, China,  <sup>3</sup> Key Laboratory of Polar Science, MNR, Polar Research Institute of China, Shanghai, China.

Paper: ([https://doi.org/10.1038/s41612-025-01058-0](https://www.nature.com/articles/s41612-025-01058-0))

</div>



# <font style="color:rgb(64, 64, 64);">IceMamba</font>
**<font style="color:rgb(64, 64, 64);">IceMamba: Arctic Sea Ice Forecasting Framework</font>**<font style="color:rgb(64, 64, 64);">  
</font><font style="color:rgb(64, 64, 64);">Developed through </font>**<font style="color:rgb(64, 64, 64);">multi-institutional collaboration</font>**<font style="color:rgb(64, 64, 64);"> led by the School of </font>**<font style="color:rgb(64, 64, 64);">Computer Science at Fudan University</font>**<font style="color:rgb(64, 64, 64);">, in partnership with the Department of Atmospheric Sciences and China Polar Research Institute, IceMamba is an open-source deep learning framework specializing in </font>**<font style="color:rgb(64, 64, 64);">pan-Arctic seasonal sea ice concentration (SIC) forecasting</font>**<font style="color:rgb(64, 64, 64);">.</font>

### <font style="color:rgb(64, 64, 64);">📦</font><font style="color:rgb(64, 64, 64);"> Open Science Resources</font>
<font style="color:rgb(64, 64, 64);">We provide full accessibility to accelerate polar climate research:</font>

+ **<font style="color:rgb(64, 64, 64);">Model Weights</font>**<font style="color:rgb(64, 64, 64);">: Pretrained parameters for rapid deployment</font>
+ **<font style="color:rgb(64, 64, 64);">Training Datasets</font>**<font style="color:rgb(64, 64, 64);">: Processed SIC observations with spatiotemporal metadata</font>
+ **<font style="color:rgb(64, 64, 64);">Configuration Files</font>**<font style="color:rgb(64, 64, 64);">: Pre-optimized experimental setups</font>

<font style="color:rgb(64, 64, 64);">🔗</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">Persistent Access</font>**<font style="color:rgb(64, 64, 64);">:  
</font><font style="color:rgb(64, 64, 64);">All assets are permanently hosted on </font>[Zenodo](https://zenodo.org/records/xxxxxx)<font style="color:rgb(64, 64, 64);"> with version control：</font>[https://zenodo.org/records/14926245](https://zenodo.org/records/14926245)

<font style="color:rgb(64, 64, 64);">📜</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">Licensing</font>**<font style="color:rgb(64, 64, 64);">:</font>

+ <font style="color:rgb(64, 64, 64);">Data: CC-BY 4.0 International</font>
+ <font style="color:rgb(64, 64, 64);">Code: MIT License</font>

# Get start
```bash
git https://github.com/WeiWang31/IceMamba.git
cd IceMamba
```

## Installation
**<font style="color:#DF2A3F;">The base environment is cuda 11.7=11.7, python=3.7</font>**

1. create conda environment

```bash
conda create -n icemamba python=3.7
conda activate icemamba
```

2. Install Dependencies

```bash
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.9.5
pip install torchsummary
cd selective_scan && pip install . && pytest
```

### <font style="color:rgb(64, 64, 64);">Data Acquisition</font>
<font style="color:rgb(64, 64, 64);">🔗</font><font style="color:rgb(64, 64, 64);"> Download the complete resource bundle from our Zenodo repository:  
</font>_<font style="color:rgb(64, 64, 64);">Package includes</font>_<font style="color:rgb(64, 64, 64);">:</font>

+ <font style="color:rgb(64, 64, 64);">Raw observational datasets</font>
+ <font style="color:rgb(64, 64, 64);">Pretrained model checkpoints (CKPT)</font>

```bash
# Unpack archives and configure paths
unzip ERA5_EASE.zip
unzip ORAS5_EASE.zip
unzip ckpt.zip
unzip IceMamba-4_SIPN_ckpt.zip
unzip IceMamba-1-only-SIC_SIPN_ckpt.zip

# Data deployment
mv ERA5_EASE ORAS5_EASE ./data_preprocess/

# Model deployment
mv ckpt/ ./Model/IceMamba/
mv IceMamba-4_SIPN_ckpt/ ./SIPN_evaluation_IceMamba-4
mv IceMamba-1-only-SIC_SIPN_ckpt/ ./SIPN_evaluation_IceMamba-1-only-SIC
```

## Date Preprocess
```bash
cd ./data_preprocess
python data_preprocess.py
```

## Test models
1. Testing all IceMamba variants

```bash
cd ./Model/IceMamba
bash test.sh
```

2. Testing the IceMamba-4 for SIPN benchmark

```bash
cd ./SIPN_evaluation_IceMamba-4
bash SIPN_evaluation.sh
python calculate_SIPN_evaluation_result.py
```

3. Testing the IceMamba-1-only-SIC for SIPN benchmark

```bash
cd ./SIPN_evaluation_IceMamba-1-only-SIC
python IceMamba-1-only_SIC_test.py
python calculate_SIPN_evaluation_result.py
```

## Explainability Test

```bash
cd ./explainability
# generate two test sets
python generate_testset.py
# Explainability Test
python permute_test.py
# calculate MAE after permuting
python permute_result_analysis_month.py
python permute_result_analysis_month_ua10.py
```


## Training models
If you want to train IceMamba variants like in our paper. You can:

```bash
cd ./Model/IceMamba
bash train_iceMamba-1.sh
bash train_iceMamba-4.sh
bash train_iceMamba-6.sh
```

If you want to train IceMamba-4 for SIPN benchmark like in our paper. You can:

```bash
cd ./SIPN_evaluation_IceMamba-4
bash train.sh
```

If you want to train IceMamba-1-only-SIC for SIPN benchmark like in our paper. You can:

```bash
cd ./SIPN_evaluation_IceMamba-1-only-SIC
bash train.sh
```

## **<font style="color:rgb(64, 64, 64);">Custom Model Training (Optional)</font>**<font style="color:rgb(64, 64, 64);"></font>
<font style="color:rgb(64, 64, 64);">Leverage our framework to develop novel IceMamba variants or entirely new sea ice prediction architectures. Implementation workflow:</font>

<font style="color:rgb(64, 64, 64);">🛠</font><font style="color:rgb(64, 64, 64);"> Configuration Guide</font>

1. **<font style="color:rgb(64, 64, 64);">Preprocessing Control</font>**<font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">(</font><font style="color:rgb(64, 64, 64);">preprocess_dict_xx.json</font><font style="color:rgb(64, 64, 64);">)</font>
    - <font style="color:rgb(64, 64, 64);">"abs"</font><font style="color:rgb(64, 64, 64);">: Standard normalization</font>
    - <font style="color:rgb(64, 64, 64);">"anomaly"</font><font style="color:rgb(64, 64, 64);">: Anomaly-normalized hybrid processing</font>
    - _<font style="color:rgb(64, 64, 64);">Critical</font>_<font style="color:rgb(64, 64, 64);">: Variables must maintain</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">strict alignment</font>**<font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">with</font><font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">time_dict_xx.json</font>
2. **<font style="color:rgb(64, 64, 64);">Temporal Configuration</font>**<font style="color:rgb(64, 64, 64);"> </font>(`time_dict_xx.json`<font style="color:rgb(64, 64, 64);">)</font>
    - <font style="color:rgb(64, 64, 64);">Defines historical windowing strategy through</font><font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">lag_month</font><font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">parameters</font>
    - <font style="color:rgb(64, 64, 64);">Supports dynamic temporal dependency engineering</font>



<font style="color:rgb(64, 64, 64);">Overall</font>**<font style="color:rgb(64, 64, 64);">, IceMamba</font>**<font style="color:rgb(64, 64, 64);"> is not a single-purpose model, but a modular deep learning framework specifically designed for flexible sea ice prediction across multiple temporal scales (e.g., short-term, seasonal, and long-term forecasting). Key features include:</font>

+ <font style="color:rgb(64, 64, 64);">🧩</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">Decoupled Architecture</font>**<font style="color:rgb(64, 64, 64);">: Our implementation with</font><font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">PyTorch Lightning</font><font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">creates clean separation between model components and training workflow</font>
+ <font style="color:rgb(64, 64, 64);">🔧</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">Customizable Framework</font>**<font style="color:rgb(64, 64, 64);">: Researchers can either:  
</font><font style="color:rgb(64, 64, 64);">a) Modify/replace the core IceMamba model  
</font><font style="color:rgb(64, 64, 64);">b) Directly adopt our training infrastructure for custom sea ice prediction models</font>
+ <font style="color:rgb(64, 64, 64);">🤝</font><font style="color:rgb(64, 64, 64);"> </font>**<font style="color:rgb(64, 64, 64);">Research Community</font>**<font style="color:rgb(64, 64, 64);">: We actively encourage academic collaborations. Feel free to:</font>
    - <font style="color:rgb(64, 64, 64);">Build upon our codebase</font>
    - <font style="color:rgb(64, 64, 64);">Contact us at</font><font style="color:rgb(64, 64, 64);"> </font>[wang_wei23@m.fudan.edu.cn](mailto:wang_wei23@m.fudan.edu.cn)<font style="color:rgb(64, 64, 64);"> </font><font style="color:rgb(64, 64, 64);">for technical discussions</font>

<font style="color:rgb(64, 64, 64);">If this work contributes to your research, please consider leaving a </font><font style="color:rgb(64, 64, 64);">⭐</font><font style="color:rgb(64, 64, 64);">️ </font>**<font style="color:rgb(64, 64, 64);">star</font>**<font style="color:rgb(64, 64, 64);"> to support open scientific development.</font>

**<font style="color:rgb(64, 64, 64);"></font>**


## **<font style="color:rgb(64, 64, 64);">Citation</font>**<font style="color:rgb(64, 64, 64);"></font>

```bash
@article{wang2025seasonal,
  title={Seasonal forecasting of Pan-Arctic sea ice with state space model},
  author={Wang, Wei and Yang, Weidong and Wang, Lei and Wang, Guihua and Lei, Ruibo},
  journal={npj Climate and Atmospheric Science},
  volume={8},
  number={1},
  pages={1--17},
  year={2025},
  publisher={Nature Publishing Group}
}
```


## **<font style="color:rgb(64, 64, 64);">Acknowledgment</font>**<font style="color:rgb(64, 64, 64);"></font>
This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer))
