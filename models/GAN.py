import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from utils.logger import print_log

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class MODEL(pl.LightningModule):
    global G_LOSS, D_LOSS
    G_LOSS = AverageMeter()
    D_LOSS = AverageMeter()
    
    def __init__(self, config, save_dir, logger):
        super().__init__()
        self.conf = config
        self.img_size = config.input_shape[0] * config.input_shape[1] * config.input_shape[2]
        self.save_dir = save_dir
        self.print_log = logger

        self.Discriminator = nn.Sequential(
            nn.Linear(self.img_size, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(256, 1),
            nn.Sigmoid())
        
        self.Generator = nn.Sequential(
            nn.Linear(self.conf.latent_dim, 128), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(256, 512), 
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(512, 1024), 
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Linear(1024, self.img_size),
            nn.Tanh())
        
        self.criterion = nn.BCELoss()
        
    def forward(self, z):
        # in lightning, forward defines the prediction/inference actions
        output = self.Generator(z)
        output = output.reshape(z.shape[0], self.conf.input_shape[0], 
                                self.conf.input_shape[1], self.conf.input_shape[2])
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        img, _ = batch
        img = img.view(img.size(0), -1)
        self.z = torch.rand(img.size(0), self.conf.latent_dim).type_as(img)
        if optimizer_idx == 0:
            ####### Generator로 gradient가 흐르지 않도록 detach !!!!
            loss = self.criterion(self.Discriminator(img), torch.ones(img.size(0), 1).type_as(img)) + \
            self.criterion(self.Discriminator(self.Generator(self.z).detach()), torch.zeros(img.size(0), 1).type_as(img))
            loss = loss * 0.5
            D_LOSS.update(loss)
            # Logging to TensorBoard by default
            self.log("d_loss", loss)
            return loss
        
        else:
            loss = self.criterion(self.Discriminator(self.Generator(self.z)), torch.ones(img.size(0), 1).type_as(img))
            G_LOSS.update(loss)
            # Logging to TensorBoard by default
            self.log("g_loss", loss)
            return loss

    # --------------------------------------------
    # training epoch end
    # --------------------------------------------
    def training_epoch_end(self, outputs):
                
        self.epoch = self.current_epoch + 1
        
        msg = ', '.join([
                    f'epoch: {self.epoch}/{self.conf.epoch}',
                    f'discriminator loss: {D_LOSS.avg:.4f}',
                    f'generator loss: {G_LOSS.avg:.4f}'
                ])
        print_log(self.print_log, msg)
        
        D_LOSS.reset()
        G_LOSS.reset()
        
        if self.epoch % self.conf.save_interval == 0:
            save_path = self.save_dir / 'images'
            save_path = str(save_path / f'epoch-{self.epoch}.png')
            gen_img = self.conf.denormalize(self(self.z))
            torchvision.utils.save_image(gen_img.data[:25], save_path, nrow=5, normalize=True)

    def configure_optimizers(self):
        d_optimizer = torch.optim.Adam(self.Discriminator.parameters(), lr=self.conf.lr, betas=(self.conf.b1, self.conf.b2))
        g_optimizer = torch.optim.Adam(self.Generator.parameters(), lr=self.conf.lr, betas=(self.conf.b1, self.conf.b2))
        return [d_optimizer, g_optimizer], []