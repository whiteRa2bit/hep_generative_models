import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from generation.data.data_simulation import Nakagami
from generation.data.dataset_pytorch import SignalsDataset
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

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.x_dim)),
            nn.Tanh()
        )

    def forward(self, z):
        signal = self.model(z)
        return signal


class Discriminator(nn.Module):
    def __init__(self, x_dim):
        super(Discriminator, self).__init__()
        self.x_dim = x_dim

        self.model = nn.Sequential(
            nn.Linear(self.x_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal):
        validity = self.model(signal)

        return validity


def run_train(dataloader, device='cpu', **kwargs):
    generator = Generator(kwargs['sample_size'])
    discriminator = Discriminator(kwargs['sample_size'])

    generator.to(device)
    discriminator.to(device)

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

            # Print and plot every now and then
        if epoch % kwargs['print_each'] == 0:
            print(
                'epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.cpu().data.numpy(), G_loss.cpu().data.numpy()))

        kwargs['model_name'] = 'discriminator'
        save_checkpoint(discriminator, epoch, **kwargs)
        kwargs['model_name'] = 'generator'
        save_checkpoint(generator, epoch, **kwargs)

    return generator


def generate_new_signal(generator, device='cpu'):
    z = Variable(torch.randn(2, generator.latent_dim)).to(device)
    return generator(z)[0].cpu().detach().numpy()


def main():
    args = parse_args()

    # Data params
    Q_LOWER = 0.001
    Q_UPPER = 0.999
    NU_MIN = 0.9
    NU_MAX = 1.2
    NU_STEP = 0.005

    nakagami = Nakagami(args.sample_size, Q_LOWER, Q_UPPER)
    nu_values = np.arange(NU_MIN, NU_MAX, NU_STEP)
    data = nakagami.get_nakagami_data(nu_values)
    dataset = SignalsDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    run_train(dataloader, device,
              latent_dim=args.latent_dim,
              sample_size=args.sample_size,
              learning_rate=args.learning_rate,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              print_each=args.print_each,
              verbose=args.verbose,
              cpu=args.cpu,
              save_dir=args.save_dir,
              save_interval=args.save_interval,
              no_save=args.no_save,
              model_name=args.model_name)


if __name__ == '__main__':
    main()
