import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from loguru import logger


class Generator(torch.nn.Module):
    def __init__(self, config):
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']
        super().__init__()

        # self.fc = nn.Linear(self.z_dim, 16 * self.z_dim)

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
        self.block4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_channels, out_channels=9, kernel_size=4, stride=4, padding=0)
        )

        # Output shape: [batch_size, 9, 512]


    def forward(self, x, debug=False):
        def _debug():
            if debug:
                logger.info(x.shape)

        x = x.unsqueeze(2)
        _debug()
        # x = self.fc(x)
        # _debug()
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
    def visualize(generated, real, epoch):
        generated_sample = generated[0].cpu().data
        real_sample = real[0].cpu().data

        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(9):
            ax[i // 3][i % 3].plot(generated_sample[i])
        wandb.log({"Generated": fig})
        plt.clf()



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.x_dim = config['x_dim']

        # self.pool = nn.AvgPool1d(5, 3)
        # self.conv1 = nn.Conv1d(9, 16, 7, padding=3)
        # self.conv2 = nn.Conv1d(16, 8, 5, padding=2)
        # # self.conv3 = nn.Conv1d(32, 8, 5, padding=2)

        # layernorm_dim = config["x_dim"]
        # self.layernorm1 = nn.LayerNorm([16, layernorm_dim])
        # layernorm_dim = (layernorm_dim - 2) // 3
        # self.layernorm2 = nn.LayerNorm([8, layernorm_dim])
        # layernorm_dim = (layernorm_dim - 2) // 3
        # self.layernorm3 = nn.LayerNorm([8, layernorm_dim])
        # layernorm_dim = (layernorm_dim - 2) // 3

        self.fc1 = nn.Linear(self.x_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc_final = nn.Linear(9, 1)

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

# class Discriminator(nn.Module):
#     def __init__(self, config):
#         super(Discriminator, self).__init__()
#         self.x_dim = config['x_dim']

#         self.fc1 = nn.Linear(self.x_dim, 64)
#         self.fc2 = nn.Linear(64, 8)
#         self.fc_final = nn.Linear(8 * 9, 1)

#         # Input shape: [batch_size, 9, 512]
#         out_channels = config["channels"] // 2**3
#         self.block1 = nn.Sequential(
#             nn.Conv1d(in_channels=9, out_channels=out_channels, kernel_size=4, stride=4, padding=0),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         # Input shape: [batch_size, channels/8, 128]
#         self.block2 = nn.Sequential(
#             nn.Conv1d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=4, stride=4, padding=0),
#             # nn.LayerNorm(512),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         out_channels *= 2

#         # Input shape: [batch_size, channels/4, 32]
#         self.block3 = nn.Sequential(
#             nn.Conv1d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=4, stride=4, padding=1),
#             # nn.LayerNorm(1024),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         out_channels *= 2

#         # Input shape: [batch_size, channels/2, 8]
#         self.block4 = nn.Sequential(
#             nn.Conv1d(in_channels=out_channels, out_channels=1, kernel_size=8, stride=1, padding=0)
#         )

#     def forward(self, x, debug=False):
#         def _debug():
#             if debug:
#                 logger.info(x.shape)

#         x = self.block1(x)
#         _debug()
#         x = self.block2(x)
#         _debug()
#         x = self.block3(x)
#         _debug()
#         x = self.block4(x)
#         _debug()
#         x = x.squeeze(1)
#         _debug()
        
#         # x = torch.tanh(self.fc1(x))
#         # _debug()
#         # x = torch.tanh(self.fc2(x))
#         # _debug()
#         # x = x.view(x.shape[0], -1)
#         # _debug()
#         # x = self.fc_final(x)
#         # _debug()

#         return x
