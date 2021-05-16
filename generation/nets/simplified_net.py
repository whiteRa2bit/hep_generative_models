import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator
from generation.config import TIME_NORM_COEF
from generation.metrics.amplitude_metrics import get_space_metrics_dict, get_amplitude_fig
from generation.metrics.time_metrics import get_time_values, plot_time_distributions
from generation.metrics.utils import calculate_1d_distributions_distances, get_correlations

class SimplifiedGenerator(AbstractGenerator):
    def __init__(self, config):
        super(SimplifiedGenerator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.final = nn.Sequential(
            nn.Linear(self.z_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.x_dim // 2),
            nn.ReLU(),
            nn.Tanh()
        )

        self.W1 = torch.nn.Parameter(torch.randn(self.x_dim // 2))
        self.W2 = torch.nn.Parameter(torch.randn(self.x_dim // 2))
        self.W1.requires_grad = True
        self.W2.requires_grad = True

        # self.amplitude_head = nn.Sequential(
        #     nn.Linear(self.x_dim // 2, self.x_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.x_dim, self.x_dim // 2),
        #     nn.Tanh()
        # )

        # self.time_head = nn.Sequential(
        #     nn.Linear(self.x_dim // 2, self.x_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(self.x_dim, self.x_dim // 2),
        # )

    def forward(self, x, debug=False):
        amplitudes_features = self.final(x)
        # out_reshaped = out.view(x.shape[0], 2, -1)

        # time_features = out_reshaped[:, 0]
        # amplitude_features = out_reshaped[:, 1]
        # time_out = self.time_head(time_features)
        # amplitude_out = self.amplitude_head(amplitude_features)

        # return torch.cat([time_out, amplitude_out], dim=1)
        time_features = torch.tanh(torch.tanh(amplitudes_features * self.W1) * self.W2)
        return torch.cat([time_features, amplitudes_features], dim=1)

    @staticmethod
    def get_rel_fake_fig(real_sample, fake_sample):
        real_sample = real_sample.cpu().data
        fake_sample = fake_sample.cpu().data

        plt.clf()
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        ax[0][0].set_title("Real times")
        ax[0][0].plot(real_sample[:9])
        ax[0][1].set_title("Fake times")
        ax[0][1].plot(fake_sample[:9])
        ax[1][0].set_title("Real amplitudes")
        ax[1][0].plot(real_sample[9:])
        ax[1][1].set_title("Fake amplitudes")
        ax[1][1].plot(fake_sample[9:])
        return fig

    @staticmethod
    def get_metrics_to_log(real_sample, fake_sample):
        """
        real_sample: [batch_size, 2 * detectors_num]
        fake_sample: [batch_size, 2 * detectors_num]
        """
        def get_times_amplitudes(items):
            times = np.array([item[:9] for item in items]).T
            amplitudes = np.array([item[9:] for item in items]).T
            return times, amplitudes
        
        real_sample = real_sample.cpu().detach().numpy()
        fake_sample = fake_sample.cpu().detach().numpy()

        real_times, real_amplitudes = get_times_amplitudes(real_sample)
        fake_times, fake_amplitudes = get_times_amplitudes(fake_sample)

        real_times *= TIME_NORM_COEF
        fake_times *= TIME_NORM_COEF

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
        time_dict['Time distribution'] = wandb.Image(time_fig)

        return {**time_dict, **amplitude_dict}


class SimplifiedDiscriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(SimplifiedDiscriminator, self).__init__(config)
        self.x_dim = config['x_dim']

        self.fc_final = nn.Sequential(
            nn.Linear(self.x_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, x, debug=False):
        out = self.fc_final(x)
        return out
