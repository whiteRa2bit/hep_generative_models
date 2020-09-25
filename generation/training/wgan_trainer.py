import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable, grad
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

        for epoch in range(self.config['epochs_num']):
            for it, data in enumerate(dataloader):
                if it == dataloader.dataset.__len__() // self.config['batch_size']:
                    break

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

                    alpha = torch.rand((self.config["batch_size"],
                                        1)).to(self.config['device'])  # TODO: (@whiteRa2bit, 2020-09-25) Fix shape
                    x_hat = alpha * X.data + (1 - alpha) * g_sample.data
                    x_hat.requires_grad = True
                    pred_hat = self.discriminator(x_hat)
                    gradients = grad(
                        outputs=pred_hat,
                        inputs=x_hat,
                        grad_outputs=torch.ones(pred_hat.size()).to(self.config['device']),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
                    gradient_penalty = self.config['lambda'] * (
                        (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

                    d_loss = torch.mean(d_fake) - torch.mean(d_real)
                    d_loss_gp = d_loss + gradient_penalty
                    d_loss_gp.backward()
                    self.d_optimizer.step()

                    # Housekeeping - reset gradient
                    self.reset_grad()

                if (it + 1) % self.config['log_each'] == 0:
                    wandb.log({"D loss": d_loss.cpu(), "D loss GP": d_loss_gp.cpu(), "G loss": g_loss.cpu()})

                    generated_sample = g_sample[0].cpu().data
                    real_sample = X[0].cpu().data
                    plt.title("Generated")
                    plt.plot(generated_sample)
                    wandb.log({"sample_plot": wandb.Image(plt)})
                    plt.show()
