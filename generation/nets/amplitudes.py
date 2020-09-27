import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, self.x_dim)
        self.fc2 = nn.Linear(self.x_dim, self.x_dim)
        self.batchnorm1 = nn.BatchNorm1d(self.x_dim)

    def forward(self, x, debug=False):
        x = self.fc1(x)
        x = F.leaky_relu(self.batchnorm1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    @staticmethod
    def visualize(generated, real, epoch):
        generated_sample = generated[0].cpu().data
        real_sample = real[0].cpu().data

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].set_title("Generated")
        ax[0].plot(generated_sample)
        ax[1].set_title("Real")
        ax[1].plot(real_sample)
        wandb.log({"generated_real": fig})
        plt.clf()


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.x_dim = config['x_dim']

        self.fc1 = nn.Linear(self.x_dim, self.x_dim)
        self.fc2 = nn.Linear(self.x_dim, 1)
        self.layernorm1 = nn.LayerNorm([self.x_dim])

    def forward(self, x, debug=False):
        x = self.fc1(x)
        x = F.leaky_relu(self.layernorm1(x))
        x = self.fc2(x)
        return x
