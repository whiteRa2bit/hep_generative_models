import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator
from generation.metrics.amplitude_metrics import get_space_metrics_dict, get_amplitude_fig, get_amplitude_correlations
from generation.metrics.time_metrics import get_time_values, plot_time_distributions
from generation.metrics.utils import calculate_1d_distributions_distances


class SignalsGenerator(AbstractGenerator):
    def __init__(self, config):
        super(SignalsGenerator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        # Input shape: [batch_size, z_dim, 1]
        out_channels = config["channels"]
        assert out_channels % 2 ** 3 == 0
        self.block1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=config["z_dim"], out_channels=out_channels, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=True))

        # Input shape: [batch_size, channels, 4]
        self.block2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=out_channels, out_channels=out_channels // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels // 2),
            nn.ReLU(inplace=True))
        out_channels //= 2

        # Input shape: [batch_size, channels/2, 16]
        self.block3 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=out_channels, out_channels=out_channels // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels // 2),
            nn.ReLU(inplace=True))
        out_channels //= 2

        # Input shape: [batch_size, channels/4, 64]
        assert config["pad_size"] % 2 == 1
        self.block4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm1d(num_features=out_channels // 2),
            nn.ReLU(inplace=True)
            # nn.AvgPool1d(config["pad_size"], stride=1, padding=config["pad_size"] // 2)
        )
        out_channels //= 2

        # Input shape: [batch_size, channels/8, 256]
        self.block5 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=out_channels, out_channels=9, kernel_size=4, stride=4, padding=0),
        )
        # Output shape: [batch_size, 9, 1024]

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
        x = self.block5(x)
        _debug()

        return x

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

        real_amplitude_corrs = get_amplitude_correlations(real_amplitudes)
        fake_amplitude_corrs = get_amplitude_correlations(fake_amplitudes)
        corrs_distance = np.mean(np.abs(real_amplitude_corrs - fake_amplitude_corrs))

        amplitude_dict = {
            f"Amplitude distance {detector + 1}": amplitude_distances[detector] for detector in range(len(amplitude_distances))
        }
        amplitude_dict["Amplitude correlations distance"] = corrs_distance
        amplitude_dict["Amplitudes distributions"] = wandb.Image(amplitude_fig)
        amplitude_dict = {**amplitude_dict, **space_metrics_dict}

        real_times = np.array([get_time_values(detector_sample, to_postprocess=False) for detector_sample in real_sample])
        fake_times = np.array([get_time_values(detector_sample, to_postprocess=False) for detector_sample in fake_sample])
        
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

        time_dict['Time distribution'] = wandb.Image(time_fig),

        return {**time_dict, **amplitude_dict}


class SignalsDiscriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(SignalsDiscriminator, self).__init__(config)
        x_dim = config['x_dim']
        out_channels = config['channels']

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=out_channels, kernel_size=4, stride=4, padding=0),
            nn.LayerNorm([out_channels, x_dim // 4]),
            nn.LeakyReLU(inplace=True)
        )
        x_dim //= 4

        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=4, stride=4, padding=0),
            nn.LayerNorm([out_channels * 2, x_dim // 4]),
            nn.LeakyReLU(inplace=True)
        )
        out_channels *= 2
        x_dim //= 4
        
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=4, stride=4, padding=0),
            nn.LayerNorm([out_channels * 2, x_dim // 4]),
            nn.LeakyReLU(inplace=True)
        )
        out_channels *= 2
        x_dim //= 4
        
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=4, stride=4, padding=0),
            nn.LayerNorm([out_channels * 2, x_dim // 4]),
            nn.LeakyReLU(inplace=True)
        )
        out_channels *= 2
        x_dim //= 4
        
        self.block5 = nn.Conv1d(in_channels=out_channels, out_channels=1, kernel_size=4, stride=4, padding=0)
        
    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
