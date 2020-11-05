import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.x_dim = config['x_dim']
        self.z_dim = config['z_dim']

        self.fc0 = nn.Linear(self.z_dim, self.z_dim * 9)
        self.fc1 = nn.Linear(self.z_dim, self.x_dim // 16)
        self.fc2 = nn.Linear(self.x_dim // 16, self.x_dim // 4)
        self.fc3 = nn.Linear(self.x_dim // 4, self.x_dim + 9)
        
        self.batchnorm0 = nn.BatchNorm1d(9)
        self.batchnorm1 = nn.BatchNorm1d(9)
        self.batchnorm2 = nn.BatchNorm1d(9)

        self.pool = nn.AvgPool1d(10, stride=1)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        x = self.fc0(x)
        _debug()
        x = x.view(-1, 9, self.z_dim)
        _debug()
        x = self.batchnorm0(x)
        _debug()
        x = self.batchnorm1(self.fc1(x))
        x = torch.tanh(x)
        _debug()
        x = self.batchnorm2(self.fc2(x))
        x = torch.tanh(x)
        _debug()
        x = torch.sigmoid(self.fc3(x))
        _debug()
        x = self.pool(x)
        _debug()

        return x


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
        self.fc2 = nn.Linear(64, 8)
        self.fc_final = nn.Linear(8 * 9, 1)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        x = torch.tanh(self.fc1(x))
        _debug()
        x = torch.tanh(self.fc2(x))
        _debug()
        x = x.view(x.shape[0], -1)
        _debug()
        x = self.fc_final(x)

        return x
