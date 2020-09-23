import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, x_dim, latent_dim=100):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.in_channels = 16

        self.fc1 = nn.Linear(self.latent_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, self.in_channels * self.x_dim)

        self.conv1 = nn.Conv1d(self.in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 4, 3, padding=1)
        self.conv3 = nn.Conv1d(4, 1, 3, padding=1)

    def forward(self, z):
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = out.view(out.shape[0], self.in_channels, self.x_dim)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)

        return F.sigmoid(out.squeeze(1))


class Discriminator(nn.Module):
    def __init__(self, x_dim):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim
        self.in_channels = 16

        self.fc1 = nn.Linear(self.x_dim, self.x_dim * self.in_channels)
        self.fc_final = nn.Linear(self.x_dim, 1)

        self.conv1 = nn.Conv1d(self.in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv1d(8, 4, 3, padding=1)
        self.conv3 = nn.Conv1d(4, 1, 3, padding=1)

    def forward(self, signal):
        out = F.relu(self.fc1(signal))

        out = out.view(out.shape[0], self.in_channels, self.x_dim)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = out.squeeze(1)
        out = self.fc_final(out)

        return out