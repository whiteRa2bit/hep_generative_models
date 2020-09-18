import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import wandb
wandb.init(project="hep_generative_models")

from generation.dataset.dataset_pytorch import SignalsDataset
from generation.train.utils import save_checkpoint, parse_args


class Generator(nn.Module):
    def __init__(self, x_dim, latent_dim=100):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*block(self.latent_dim, 16, normalize=False), *block(16, 32), *block(32, 64),
                                   nn.Linear(64, int(self.x_dim)), nn.Sigmoid())

    def forward(self, z):
        signal = self.model(z)
        return signal


class Discriminator(nn.Module):
    def __init__(self, x_dim):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim

        self.model = nn.Sequential(
            nn.Linear(self.x_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal):
        validity = self.model(signal)

        return validity


def run_train(dataset, generator_class=None, discriminator_class=None, **kwargs):
    device = 'cpu' if kwargs['cpu'] else 'cuda'
    dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True)

    if generator_class is None:
        generator_class = Generator
    if discriminator_class is None:
        discriminator_class = Discriminator
    generator = generator_class(x_dim=kwargs['sample_size'], latent_dim=kwargs['latent_dim'])
    discriminator = discriminator_class(kwargs['sample_size'])

    generator.to(device)
    discriminator.to(device)

    wandb.watch(generator)
    wandb.watch(discriminator)

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=kwargs['learning_rate'])
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=kwargs['learning_rate'])

    ones_label = Variable(torch.ones(kwargs['batch_size'], 1)).to(device)
    zeros_label = Variable(torch.zeros(kwargs['batch_size'], 1)).to(device)

    for epoch in range(kwargs['num_epochs']):
        for it, signal in enumerate(dataloader):
            if signal.shape[0] != kwargs['batch_size']:
                break

            # Train Generator
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            z = Variable(torch.randn(kwargs['batch_size'], kwargs['latent_dim'])).to(device)
            X = Variable(signal).to(device)
            G_sample = generator(z)
            D_fake = discriminator(G_sample)

            G_loss = F.binary_cross_entropy(D_fake, ones_label)

            G_loss.backward(retain_graph=True)
            G_optimizer.step()

            # Train discriminator
            D_optimizer.zero_grad()
            G_optimizer.zero_grad()

            D_real = discriminator(X)

            D_loss_real = F.binary_cross_entropy(D_real, ones_label)
            D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
            D_loss = D_loss_real + D_loss_fake

            D_loss.backward()
            D_optimizer.step()

        # Log and plot
        if kwargs['verbose'] and epoch % kwargs['print_each'] == 0:
            wandb.log({"D loss": D_loss.cpu().data.numpy(), "G loss": G_loss.cpu().data.numpy()})

            rows_num = 3
            samples = generator(z).cpu().data.numpy()[:rows_num ** 2]

            f, ax = plt.subplots(rows_num, rows_num, figsize=(rows_num ** 2, rows_num ** 2))
            gs = gridspec.GridSpec(rows_num, rows_num)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax[i // rows_num][i % rows_num].plot(sample)
            plt.show()

        kwargs['model_name'] = 'discriminator'
        save_checkpoint(discriminator, epoch, **kwargs)
        kwargs['model_name'] = 'generator'
        save_checkpoint(generator, epoch, **kwargs)

    torch.save(generator.state_dict(), os.path.join(wandb.run.dir, 'generator.pt'))
    torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, 'discriminator.pt'))

    return generator


def generate_new_signal(generator, device='cpu', signals_num=1):
    generator.to(device)
    z = Variable(torch.randn(signals_num + 1, generator.latent_dim)).to(device)
    return generator(z)[:signals_num].cpu().detach().numpy()
