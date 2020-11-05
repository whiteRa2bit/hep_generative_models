import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from loguru import logger


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_dim = config["z_dim"]

        self.fc1 = nn.Linear(self.z_dim, 400)

        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.batchnorm4 = nn.BatchNorm2d(8)

        self.conv1 = nn.ConvTranspose2d(4, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.conv5 = nn.Conv2d(8, 4, 3)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

        _debug()
        x = F.leaky_relu(self.fc1(x))
        _debug()

        x = x.view(-1, 4, 10, 10)
        _debug()
        x = F.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        _debug()
        x = F.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        _debug()
        x = F.leaky_relu(self.conv3(x))
        x = self.batchnorm3(x)
        _debug()
        x = F.leaky_relu(self.conv4(x))
        x = self.batchnorm4(x)
        _debug()
        x = F.leaky_relu(self.conv5(x))
        _debug()
        return x

    @staticmethod
    def visualize(generated, real, epoch):
        generated_sample = generated[0].cpu().data
        real_sample = real[0].cpu().data

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].set_title("Generated")
        ax[0].imshow(generated_sample)
        ax[1].set_title("Real")
        ax[1].imshow(real_sample)
        wandb.log({"generated_real": fig})
        plt.clf()


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.pool = nn.AvgPool2d(5, 3)
        self.conv1 = nn.Conv2d(4, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 7 * 7, 1)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        _debug()
        x = F.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        _debug()
        x = self.pool(x)
        _debug()
        x = F.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        _debug()
        x = self.pool(x)
        _debug()
        x = F.leaky_relu(self.conv3(x))
        x = self.batchnorm3(x)
        _debug()

        x = x.view(-1, 16 * 7 * 7)
        _debug()
        x = self.fc1(x)
        _debug()
        return x
