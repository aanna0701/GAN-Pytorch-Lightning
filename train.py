from dataloader.call_data import DATASET
from pytorch_lightning import Trainer
import argparse
import sys
import importlib
from utils.logger import print_log
import time
from pathlib import Path

now = time.localtime()
SAVE_DIR = Path('save') / f'{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m-{now.tm_sec}s'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = str(SAVE_DIR / 'log.txt')
# =========================================== Arguments ===========================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='train the face recognition network')
    parser.add_argument('--config', default="config_vanila", help='name of config file without file extension')
    
    args = parser.parse_args()

    return args

# ==================================================================================================

def main():
    
    
    # --------------------------------------------
    # train arguments
    # --------------------------------------------
    global args
    args = parse_args()

    # --------------------------------------------
    # train configurations
    # --------------------------------------------   
    config = importlib.import_module(f"configs.{args.config}")
    global conf
    conf = config.conf

    msg_conf = "*"*30 + ' CONFIG ' + "*"*30 + "\n"
    for k in conf: msg_conf += f"{k}: {conf[k]}" + "\n"
    msg_conf += "*"*30 + ' CONFIG ' + "*"*30  
    print_log(LOGGER, msg_conf)
    del msg_conf

    dm = DATASET[conf.dataset]()
    print(conf.network)
    model_module = importlib.import_module(f"models.{conf.network}")
    model = model_module.MODEL(conf, SAVE_DIR, LOGGER)

    trainer = Trainer(gpus=1, max_epochs=conf.epoch, enable_progress_bar = False)
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    main()