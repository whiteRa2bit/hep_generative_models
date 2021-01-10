import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from loguru import logger

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator


class Generator(AbstractGenerator):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        # Input shape: [batch_size, z_dim, 1]
        out_channels = config["channels"]
        assert out_channels % 2 ** 3 == 0
        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=config["z_dim"], out_channels=out_channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(inplace=True))

        # Input shape: [batch_size, channels, 4]
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=out_channels, out_channels=out_channels // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels // 2),
            nn.LeakyReLU(inplace=True))
        out_channels //= 2

        # Input shape: [batch_size, channels/2, 16]
        self.block3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=out_channels, out_channels=out_channels // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels // 2),
            nn.LeakyReLU(inplace=True))
        out_channels //= 2

        # Input shape: [batch_size, channels/4, 64]
        assert config["pad_size"] % 2 == 1
        self.block4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=9, kernel_size=4, stride=2, padding=1),
            # nn.AvgPool1d(config["pad_size"], stride=1, padding=config["pad_size"] // 2)
        )

        # Output shape: [batch_size, 9, 128]

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

        x = x.unsqueeze(2)
        _debug()
        x = self.block1(x)
        _debug()
        x = self.block2(x)
        _debug()
        x = self.block3(x)
        _debug()
        x = self.block4(x)
        _debug()

        return torch.tanh(x)

    @staticmethod
    def get_rel_fake_fig(real_sample, fake_sample):
        real_sample = real_sample.cpu().data
        fake_sample = fake_sample.cpu().data

        fig, ax = plt.subplots(3, 6, figsize=(10, 20))
        for i in range(9):  # TODO: (@whiteRa2bit, 2021-01-05) Replace with config constant
            ax[i // 3][i % 3].plot(real_sample[i])
            ax[i // 3][3 + i % 3].plot(fake_sample[i])
        return fig


class Discriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.x_dim = config['x_dim']

        self.fc1 = nn.Linear(self.x_dim, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc_final = nn.Linear(8 * 9, 1)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

        x = torch.tanh(self.fc1(x))
        _debug()
        x = torch.tanh(self.fc2(x))
        _debug()
        x = x.view(x.shape[0], -1)
        _debug()
        x = self.fc_final(x)
        _debug()

        return x
