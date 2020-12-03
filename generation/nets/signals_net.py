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
        assert out_channels % 2**3 == 0
        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=config["z_dim"], out_channels=out_channels, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
        )

        # Input shape: [batch_size, channels, 8]
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels//2),
            nn.LeakyReLU(inplace=True)
        )
        out_channels //= 2

        # Input shape: [batch_size, channels/2, 32]
        self.block3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels//2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels//2),
            nn.LeakyReLU(inplace=True)
        )
        out_channels //= 2

        # Input shape: [batch_size, channels/4, 128]
        assert config["pad_size"] % 2 == 1
        self.block4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=9, kernel_size=4, stride=4, padding=0),
            # nn.MaxPool1d(config["pad_size"], stride=1, padding=config["pad_size"]//2)
        )

        # Output shape: [batch_size, 9, 512]


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
    def visualize(generated_sample, real_sample):
        def get_figure(sample):
            fig, ax = plt.subplots(3, 3, figsize=(10, 10))
            for i in range(9):
                ax[i // 3][i % 3].plot(sample[i])
            return fig
        
        generated_sample = generated_sample.cpu().data
        real_sample = real_sample.cpu().data
        fig_gen = get_figure(generated_sample)
        fig_real = get_figure(real_sample)
        wandb.log({"generated": fig_gen, "real": fig_real})
        plt.clf()


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

        return x
