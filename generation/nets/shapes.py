import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from loguru import logger


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, self.x_dim)

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv1d(16, 8, 3, padding=1)
        self.conv5 = nn.Conv1d(8, 1, 3, padding=1)

        self.batchnorm1 = nn.BatchNorm1d(8)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.batchnorm4 = nn.BatchNorm1d(8)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

        x = F.leaky_relu(self.fc1(x))
        _debug()
        x = x.unsqueeze(1)
        _debug()
        x = self.conv1(x)
        x = F.leaky_relu(self.batchnorm1(x))
        _debug()
        x = self.conv2(x)
        x = F.leaky_relu(self.batchnorm2(x))
        _debug()
        x = self.conv3(x)
        x = F.leaky_relu(self.batchnorm3(x))
        _debug()
        x = self.conv4(x)
        x = F.leaky_relu(self.batchnorm4(x))
        _debug()
        x = self.conv5(x)
        _debug()

        return torch.sigmoid(x.squeeze(1))

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

        self.fc_final = nn.Linear(288, 1)

        self.pool = nn.AvgPool1d(5, 3)
        self.conv1 = nn.Conv1d(1, 8, 7, padding=3)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 8, 3, padding=1)

        self.layernorm1 = nn.LayerNorm([8, 1024])
        self.layernorm2 = nn.LayerNorm([32, 340])
        self.layernorm3 = nn.LayerNorm([8, 112])

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        x = x.unsqueeze(1)
        _debug()
        x = self.conv1(x)
        x = F.leaky_relu(self.layernorm1(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.conv2(x)
        x = F.leaky_relu(self.layernorm2(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = self.conv3(x)
        x = F.leaky_relu(self.layernorm3(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = x.view(x.shape[0], -1)

        x = x.squeeze(1)
        _debug()
        x = self.fc_final(x)
        _debug()

        return x
