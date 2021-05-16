import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator
from generation.metrics.amplitude_metrics import get_space_metrics_dict, get_amplitude_fig
from generation.metrics.time_metrics import get_time_values, plot_time_distributions
from generation.metrics.utils import calculate_1d_distributions_distances, get_correlations


class SignalsGenerator(AbstractGenerator):
    def __init__(self, config):
        super(SignalsGenerator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, self.x_dim)

        self.conv1 = nn.Conv1d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv1d(16, 8, 3, padding=1)
        self.conv5 = nn.Conv1d(8, 9, 3, padding=1)

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

        fig, ax = plt.subplots(3, 6, figsize=(10, 20))
        for i in range(9):  # TODO: (@whiteRa2bit, 2021-01-05) Replace with config constant
            ax[i // 3][i % 3].plot(real_sample[i])
            ax[i // 3][3 + i % 3].plot(fake_sample[i])
        return fig

    @staticmethod
    def get_metrics_to_log(real_sample, fake_sample):
        """
        real_sample: [batch_size, detectors_num, signal_size]
        fake_sample: [batch_size, detectors_num, signal_size]
        """
        real_sample = np.transpose(real_sample.cpu().detach().numpy(), (1, 0, 2))
        fake_sample = np.transpose(fake_sample.cpu().detach().numpy(), (1, 0, 2))

        real_amplitudes = np.max(real_sample, axis=2) - np.min(real_sample, axis=2)
        fake_amplitudes = np.max(fake_sample, axis=2) - np.min(fake_sample, axis=2)

        space_metrics_dict = get_space_metrics_dict(real_amplitudes, fake_amplitudes)
        amplitude_distances = calculate_1d_distributions_distances(real_amplitudes, fake_amplitudes)
        amplitude_fig = get_amplitude_fig(real_amplitudes, fake_amplitudes)

        real_amplitude_corrs = get_correlations(real_amplitudes)
        fake_amplitude_corrs = get_correlations(fake_amplitudes)
        amplitude_corrs_distance = np.mean(np.abs(real_amplitude_corrs - fake_amplitude_corrs))

        amplitude_dict = {
            f"Amplitude distance {detector + 1}": amplitude_distances[detector] for detector in range(len(amplitude_distances))
        }
        amplitude_dict["Amplitude correlations distance"] = amplitude_corrs_distance
        amplitude_dict["Amplitudes distributions"] = wandb.Image(amplitude_fig)
        amplitude_dict = {**amplitude_dict, **space_metrics_dict}

        real_times = np.array([get_time_values(detector_sample, to_postprocess=False) for detector_sample in real_sample])
        fake_times = np.array([get_time_values(detector_sample, to_postprocess=False) for detector_sample in fake_sample])
        
        real_times_corrs = get_correlations(real_times)
        fake_times_corrs = get_correlations(fake_times)
        time_corrs_distance = np.mean(np.abs(real_times_corrs - fake_times_corrs))

        time_distances = calculate_1d_distributions_distances(real_times, fake_times)
        time_dict = {
            f"Time distance {detector + 1}": time_distances[detector] for detector in range(len(time_distances))
        }

        time_fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        time_fig.suptitle("Times distributions", fontsize=16)
        for i in range(9):
            plot_time_distributions(
                real_times=real_times[i], 
                fake_times=fake_times[i], 
                ax=ax[i // 3][i % 3], 
                title=f'Detector {i + 1}',
                bins=[x for x in np.arange(0, 200, 10)]
            )
        time_dict["Time correlations distance"] = time_corrs_distance
        time_dict['Time distribution'] = wandb.Image(time_fig),

        return {**time_dict, **amplitude_dict}


class SignalsDiscriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(SignalsDiscriminator, self).__init__(config)
        self.x_dim = config['x_dim']

        self.fc_final = nn.Linear(288, 1)

        self.pool = nn.AvgPool1d(5, 3)
        self.conv1 = nn.Conv1d(9, 8, 7, padding=3)
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
        # x = x.unsqueeze(1)
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
