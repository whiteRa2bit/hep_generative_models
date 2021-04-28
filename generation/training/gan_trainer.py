import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loguru import logger

from generation.training.abstract_trainer import AbstractTrainer


class GanTrainer(AbstractTrainer):
    def run_train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        self._initialize_wandb()

        criterion = F.binary_cross_entropy
        ones_label = Variable(torch.ones(self.config['batch_size'], 1))
        ones_label = ones_label.to(self.config['device'])
        zeros_label = Variable(torch.zeros(self.config['batch_size'], 1))
        zeros_label = zeros_label.to(self.config['device'])

        for epoch in tqdm.tqdm(range(self.config['epochs_num'])):
            epoch_d_loss = 0
            epoch_g_loss = 0

            for it, data in enumerate(dataloader):
                if it == len(dataloader.dataset) // self.config['batch_size']:
                    break

                X = Variable(data)
                X = X.to(self.config['device'])
                z = Variable(torch.randn(self.config['batch_size'], self.config['z_dim']))
                z = z.to(self.config['device'])

                g_sample = self.generator(z)
                d_fake = self.discriminator(g_sample)

                # Train discriminator
                d_real = self.discriminator(X)

                d_loss_real = criterion(d_real, ones_label)
                d_loss_fake = criterion(d_fake, zeros_label)
                d_loss = d_loss_real + d_loss_fake
                epoch_d_loss += d_loss.item()

                d_loss.backward()
                self.d_optimizer.step()
                self._reset_grad()

                # Train Generator
                if it % self.config['d_coef'] == 0:
                    X = Variable(data)  # TODO: Do I have to generate data once more?
                    X = X.to(self.config['device'])
                    z = Variable(torch.randn(self.config['batch_size'], self.config['z_dim']))
                    z = z.to(self.config['device'])

                    g_sample = self.generator(z)
                    d_fake = self.discriminator(g_sample)
                    g_loss = criterion(d_fake, ones_label)
                    epoch_g_loss += g_loss.item()
                    g_loss.backward()
                    self.g_optimizer.step()
                    self._reset_grad()


            epoch_d_loss = epoch_d_loss / len(dataloader.dataset)
            epoch_g_loss = (epoch_g_loss * self.config['d_coef']) / len(dataloader.dataset)

            if epoch % self.config['log_each'] == 0:
                real_fake_fig = self.generator.get_rel_fake_fig(X[0], g_sample[0])
                wandb.log({
                    "D loss": d_loss.cpu().data, 
                    "G loss": g_loss.cpu().data,
                    "Real vs Fake": real_fake_fig
                })
                plt.close("all")

            if epoch % self.config['save_each'] == 0:
                self._save_checkpoint(self.generator, f"generator_{epoch}")
                self._save_checkpoint(self.discriminator, f"discriminator_{epoch}")
