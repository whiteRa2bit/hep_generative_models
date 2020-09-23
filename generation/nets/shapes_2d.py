import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_dim = config["z_dim"]
        
        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 784)
        
        self.conv1 = nn.ConvTranspose2d(4, 16, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.conv4 = nn.Conv2d(16, 4, 3)
        
    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        _debug()
        x = F.leaky_relu(self.fc1(x))
        _debug()
        x = F.leaky_relu(self.fc2(x))
        _debug()

        x = x.view(-1, 4, 14, 14)
        _debug()
        x = F.leaky_relu(self.conv1(x))
        _debug()
        x = F.leaky_relu(self.conv2(x))
        _debug()
        x = F.leaky_relu(self.conv3(x))
        _debug()
        x = F.leaky_relu(self.conv4(x))
        _debug()
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.pool = nn.MaxPool2d(5, 3)
        self.conv1 = nn.Conv2d(4, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)

        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, debug=False):
        def _debug():
            if debug:
                print(x.shape)

        _debug()
        x = F.leaky_relu(self.conv1(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = F.leaky_relu(self.conv2(x))
        _debug()
        x = self.pool(x)
        _debug()
        x = F.leaky_relu(self.conv3(x))
        _debug()

        x = x.view(-1, 16 * 2 * 2)
        _debug()
        x = F.leaky_relu(self.fc1(x))
        _debug()
        x = F.leaky_relu(self.fc2(x))
        _debug()
        return x