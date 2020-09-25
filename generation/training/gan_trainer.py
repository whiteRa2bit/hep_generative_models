import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

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

        for epoch in range(self.config['epochs_num']):
            for it, data in enumerate(dataloader):
                # Train Generator
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                X = Variable(data)
                X = X.to(self.config['device'])
                z = Variable(torch.rand(self.config['batch_size'], self.config['z_dim']))
                z = z.to(self.config['device'])

                g_sample = self.generator(z)
                d_fake = self.discriminator(g_sample)

                g_loss = criterion(d_fake, ones_label)

                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                # Train discriminator
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                d_real = self.discriminator(X)

                d_loss_real = criterion(d_real, ones_label)
                d_loss_fake = criterion(d_fake, zeros_label)
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                self.d_optimizer.step()

                if it % self.config['log_each'] == 0:
                    wandb.log({"D loss": d_loss.cpu().data, "G loss": g_loss.cpu().data})
                    self.generator.visualize(g_sample, X)
