import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import wandb

from generation.config import WANDB_PROJECT


class WganTrainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, config):
        self.generator = generator.to(config['device'])
        self.discriminator = discriminator.to(config['device'])
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.config = config

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.generator)
        wandb.watch(self.discriminator)

    def reset_grad(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def run_train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        self._initialize_wandb()

        d_loss = torch.tensor([0])  # TODO: (@whiteRa2bit, 2020-09-23) Remove
        g_loss = torch.tensor([0])  # TODO: (@whiteRa2bit, 2020-08-23) Remove
        for epoch in range(self.config['epochs_num']):
            for it, data in enumerate(dataloader):
                if it % self.config['disc_coef'] == 0:
                    # Generator forward-loss-backward-update
                    X = Variable(data)
                    X = X.to(self.config['device'])
                    z = Variable(torch.rand(self.config['batch_size'], self.config['z_dim']))
                    z = z.to(self.config['device'])

                    g_sample = self.generator(z)
                    d_fake = self.discriminator(g_sample)

                    g_loss = -torch.mean(d_fake)
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Housekeeping - reset gradient
                    self.reset_grad()
                else:
                    # Dicriminator forward-loss-backward-update
                    X = Variable(data)
                    X = X.to(self.config['device'])
                    z = Variable(torch.rand(self.config['batch_size'], self.config['z_dim']))
                    z = z.to(self.config['device'])

                    g_sample = self.generator(z)
                    d_real = self.discriminator(X)
                    d_fake = self.discriminator(g_sample)

                    d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                    d_loss.backward()
                    self.d_optimizer.step()

                    # # Weight clipping
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)  # TODO: (@whiteRa2bit, 2020-08-30) Replace with kwargs param

                    # Housekeeping - reset gradient
                    self.reset_grad()

                if it % self.config['log_each'] == 0:
                    wandb.log({"D loss": d_loss.cpu(), "G loss": g_loss.cpu()})

                    generated_sample = g_sample[0].cpu().data
                    generated_sample = np.array(generated_sample.permute(1, 2, 0))
                    generated_sample[generated_sample > 1] = 1.0
                    generated_sample[generated_sample < 0] = 0.0
                    real_sample = X[0].cpu().data
                    real_sample = np.array(real_sample.permute(1, 2, 0))

                    f, ax = plt.subplots(1, 2, figsize=(5, 12))
                    ax[0].set_title("Generated")
                    ax[0].imshow(generated_sample)
                    ax[1].set_title("Real")
                    ax[1].imshow(real_sample)
                    plt.show()
                    plt.close()
