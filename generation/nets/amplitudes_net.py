import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator
from generation.metrics.amplitude_metrics import get_space_metrics_dict, get_amplitude_fig
from generation.metrics.utils import calculate_1d_distributions_distances, get_correlations


class AmplitudesGenerator(AbstractGenerator):
    def __init__(self, config):
        super(AmplitudesGenerator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.x_dim)

    def forward(self, x, debug=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)

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
        real_sample = real_sample.T.cpu().detach().numpy()
        fake_sample = fake_sample.T.cpu().detach().numpy()

        space_metrics_dict = get_space_metrics_dict(real_sample, fake_sample)
        amplitude_distances = calculate_1d_distributions_distances(real_sample, fake_sample)
        amplitude_fig = get_amplitude_fig(real_sample, fake_sample)

        real_amplitude_corrs = get_correlations(real_amplitudes)
        fake_amplitude_corrs = get_correlations(fake_amplitudes)
        amplitude_corrs_distance = np.mean(np.abs(real_amplitude_corrs - fake_amplitude_corrs))

        amplitude_dict = {
            f"Amplitude distance {detector + 1}": amplitude_distances[detector] for detector in range(len(amplitude_distances))
        }
        amplitude_dict["Amplitude correlations distance"] = amplitude_corrs_distance
        amplitude_dict["Amplitudes distributions"] = wandb.Image(amplitude_fig)
        amplitude_dict = {**amplitude_dict, **space_metrics_dict}

        return amplitude_dict


class AmplitudesDiscriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(AmplitudesDiscriminator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc1 = nn.Linear(self.x_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x, debug=False):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
