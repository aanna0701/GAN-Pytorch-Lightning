import torch
import os
from easydict import EasyDict as edict
from pathlib import Path
################################################################################

"""
GAN model 학습에 사용되는 결과 이미지 저장 경로, 에포크 수, 모델 입력 이미지 크기 등을 정의합니다.
"""

config = edict()
config.data_path = 'data/'
config.save_path = 'save/'
config.dataset = 'MNIST'
config.epoch = 500
config.log_interval = 100
config.save_interval = 50
config.batch_size = 64
config.lr = 0.0002
config.b1 = 0.5
config.b2 = 0.999
config.input_shape = (3, 32, 32) if config.dataset == 'CIFAR10' else (1, 28, 28)
config.latent_dim = 100
config.n_workers = 4

"""
모델 입력 이미지에 수행할 normalization과 모델 생성 결과 이미지에 수행할 denormalization을 정의합니다.
"""
config.denormalize = lambda x: x*0.5+0.5
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p_data = Path(config.data_path)
p_data.mkdir(parents=True, exist_ok=True)

p_save = Path(config.save_path) / config.dataset
p_save.mkdir(parents=True, exist_ok=True)

################################################################################

################################# DEBUGGING ####################################
assert config.dataset in ['MNIST', 'CIFAR10'], 'Invalid Dataset!!!'