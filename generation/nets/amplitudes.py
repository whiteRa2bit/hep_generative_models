import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, self.x_dim)

    def forward(self, x, debug=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.clamp(x, 0, 1)
        # return torch.sigmoid(x)

    @staticmethod
    def visualize(generated, real, epoch):
        generated = generated.cpu().data
        real = real.cpu().data
        generated_sample = generated[0]
        real_sample = real[0]

        fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
        ax1[0].set_title("Generated")
        ax1[0].plot(generated_sample)
        ax1[1].set_title("Real")
        ax1[1].plot(real_sample)

        fig2, ax2 = plt.subplots(3, 3, figsize=(12, 12))
        for i in range(9):
            ax2[i // 3][i % 3].hist(real[:, i], bins=np.arange(0, 1.01, 0.1), alpha=0.7)
            ax2[i // 3][i % 3].hist(generated[:, i], bins=np.arange(0, 1.01, 0.1), alpha=0.7)
        wandb.log({"generated_real": fig1, "amplitudes distribution": wandb.Image(fig2)})
        plt.clf()
        plt.cla()


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.x_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x, debug=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
