import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from generation.nets.abstract_net import AbstractGenerator, AbstractDiscriminator


class Generator(AbstractGenerator):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.final = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.x_dim),
        )

        self.amplitude_head = nn.Sequential(
            nn.Linear(self.x_dim // 2, self.x_dim),
            nn.LeakyReLU(),
            nn.Linear(self.x_dim, self.x_dim // 2)
        )

        self.time_head = nn.Sequential(
            nn.Linear(self.x_dim // 2, self.x_dim),
            nn.Sigmoid(),
            nn.Linear(self.x_dim, self.x_dim // 2),
            nn.Sigmoid()
        )
        
        
    def forward(self, x, debug=False):
        out = self.final(x)
        out_reshaped = out.view(x.shape[0], 2, -1)
       
        time_features = out_reshaped[:, 0]
        amplitude_features = out_reshaped[:, 1]
        time_out = self.time_head(time_features)
        amplitude_out = self.amplitude_head(amplitude_features)
        
        return torch.cat([time_out, amplitude_out], dim=1)
        
        
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


class Discriminator(AbstractDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.x_dim = config['x_dim']

        # self.time_head = nn.Sequential(
        #     nn.Linear(self.x_dim // 2, 16),
        #     nn.LeakyReLU()
        # )
        
        # self.amplitude_head = nn.Sequential(
        #     nn.Linear(self.x_dim // 2, 16),
        #     nn.LeakyReLU()
        # )
        
        # self.global_head = nn.Sequential(
        #     nn.Linear(self.x_dim, 16),
        #     nn.LeakyReLU(),
        # )
        
        self.fc_final = nn.Sequential(
            nn.Linear(self.x_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )


    def forward(self, x, debug=False):
        # reshaped_x = x.view(x.shape[0], 2, -1)
        # time_features = reshaped_x[:, 0]
        # amplitude_features = reshaped_x[:, 1]
        
        # global_out = self.global_head(x)
        # time_out = self.time_head(time_features)
        # amplitude_out = self.amplitude_head(amplitude_features)
        
        # out = torch.cat([global_out, time_out, amplitude_out], dim=1)
        out = self.fc_final(x)

        
        return out
