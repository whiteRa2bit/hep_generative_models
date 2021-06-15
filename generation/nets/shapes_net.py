import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator
from generation.metrics.time_metrics import get_time_values, plot_time_distributions
from generation.metrics.utils import calculate_1d_distributions_distances


class ShapesGenerator(AbstractGenerator):
    def __init__(self, config):
        super(ShapesGenerator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']
        self._detector = config['detector']

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
    def get_rel_fake_fig(real_sample, fake_sample):
        real_sample = real_sample.cpu().data
        fake_sample = fake_sample.cpu().data

        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].set_title("Real")
        ax[0].plot(real_sample)
        ax[1].set_title("Fake")
        ax[1].plot(fake_sample)
        return fig

    @staticmethod
    def get_metrics_to_log(real_sample, fake_sample):
        """
        real_sample: [batch_size, detectors_num]
        fake_sample: [batch_size, detectors_num]
        """
        real_sample = real_sample.cpu().detach().numpy()
        fake_sample = fake_sample.cpu().detach().numpy()

        real_times = get_time_values(real_sample, to_postprocess=False)
        fake_times = get_time_values(fake_sample, to_postprocess=False)
        time_distance = calculate_1d_distributions_distances(np.array([real_times]), np.array([fake_times]))[0]

        time_fig, ax = plt.subplots(1)
        plot_time_distributions(
            real_times=real_times,
            fake_times=fake_times,
            ax=ax,
            title='Time distribution',
            bins=[x for x in np.arange(0, 200, 10)])

        time_dict = {
            'Time distance': time_distance,
            'Time distribution': wandb.Image(time_fig),
        }

        return time_dict


class ShapesDiscriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(ShapesDiscriminator, self).__init__(config)
        self.x_dim = config['x_dim']

        self.fc_final = nn.Linear(288, 1)

        self.pool = nn.AvgPool1d(5, 3)
        self.conv1 = nn.Conv1d(1, 8, 7, padding=3)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 8, 3, padding=1)

        x_dim = config["x_dim"]
        self.layernorm1 = nn.LayerNorm([8, x_dim])
        x_dim = (x_dim - 2) // 3
        self.layernorm2 = nn.LayerNorm([32, x_dim])
        x_dim = (x_dim - 2) // 3
        self.layernorm3 = nn.LayerNorm([8, x_dim])

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

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
