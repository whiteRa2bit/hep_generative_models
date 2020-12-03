import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator


class Generator(AbstractGenerator):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, (self.x_dim + self.z_dim) // 2)
        self.fc2 = nn.Linear((self.x_dim + self.z_dim) // 2, self.x_dim)
        self.fc3 = nn.Linear(self.x_dim, self.x_dim)

    def forward(self, x, debug=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.clamp(x, 0, 1)

    @staticmethod
    def visualize(generated_sample, real_sample):
        generated_sample = generated_sample.cpu().data
        real_sample = real_sample.cpu().data

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].set_title("Generated")
        ax[0].plot(generated_sample)
        ax[1].set_title("Real")
        ax[1].plot(real_sample)
        wandb.log({"generated_real": fig})
        plt.clf()


class Discriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.x_dim, self.x_dim)
        self.fc2 = nn.Linear(self.x_dim, (self.x_dim + self.z_dim) // 2)
        self.fc3 = nn.Linear((self.x_dim + self.z_dim) // 2, 1)

    def forward(self, x, debug=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
