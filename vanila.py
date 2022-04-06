from dataloader.call_data import DATASET
from models.vanila import GAN
from pytorch_lightning import Trainer
import argparse
import sys

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
    # import config file
    # --------------------------------------------
    sys.path.append("./config")
    config = __import__(args.config)
    global conf
    conf = config.config

    print("*"*30, 'CONFIG', "*"*30)
    for k in conf: print(f"{k}: {conf[k]}")
    print("*"*30, 'CONFIG', "*"*30)

    dm = DATASET[conf.dataset]()
    model = GAN(conf)

    trainer = Trainer(gpus=1, max_epochs=conf.epoch, enable_progress_bar = False)
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    main()