import torch
import os
from easydict import EasyDict as edict
from pathlib import Path
################################################################################

"""
GAN model 학습에 사용되는 결과 이미지 저장 경로, 에포크 수, 모델 입력 이미지 크기 등을 정의합니다.
"""

conf = edict()
conf.data_path = 'data/'
conf.save_path = 'save/'
conf.dataset = 'CIFAR10'
conf.epoch = 500
conf.log_interval = 100
conf.save_interval = 50
conf.batch_size = 64
conf.lr = 0.0002
conf.b1 = 0.5
conf.b2 = 0.999
conf.input_shape = (3, 32, 32) if conf.dataset == 'CIFAR10' else (1, 28, 28)
conf.latent_dim = 100
conf.n_workers = 4
conf.network = 'LSGAN'

"""
모델 입력 이미지에 수행할 normalization과 모델 생성 결과 이미지에 수행할 denormalization을 정의합니다.
"""
conf.denormalize = lambda x: x*0.5+0.5
conf.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p_data = Path(conf.data_path)
p_data.mkdir(parents=True, exist_ok=True)

p_save = Path(conf.save_path) / conf.dataset
p_save.mkdir(parents=True, exist_ok=True)

################################################################################

################################# DEBUGGING ####################################
assert conf.dataset in ['MNIST', 'CIFAR10'], 'Invalid Dataset!!!'