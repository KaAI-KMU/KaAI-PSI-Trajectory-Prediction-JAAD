The codes are based on [SGNet](https://github.com/ChuhuaW/SGNet.pytorch).


## 1. GETTING STARTED
### 1.0. Clone the repository
```buildoutcfg
git clone https://github.com/KaAI-KMU/KaAI-PSI-Trajectory-Prediction_JAAD.git
```
### 1.1. Install dependencies
Create conda environment.
```buildoutcfg
conda create -n {env_name} python=3.8
conda activate {env_name}
```
Install pytorch. Please refer to [pytorch](https://pytorch.org/get-started/locally/) for the details of installation.
```buildoutcfg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### 1.2. Download data
Please refer to [JAAD dataset](https://github.com/ykotseruba/JAAD) for the details of JAAD dataset and data structure.


You can download the pre-processed and structured data(including center optical flow) from [here]() and extract it into KaAI-PSI-Trajectory-Prediction_JAAD/data.


The optical flows are generated by using [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) model which is pretrained with Sintel.
And we use optical flow of the center of the bounding box.




## Training
### 2.1. Train the model

* Training on JAAD dataset:
```
cd KaAI-PSI-Trajectory-Prediction_JAAD
python tools/jaad/train_cvae.py --dataset JAAD --model SGNet_CVAE
```

Then, you can get a pretrained_file from KaAI-PSI-Trajectory-Prediction_JAAD/tools/jaad/checkpoints/SGNet_CVAE/1
