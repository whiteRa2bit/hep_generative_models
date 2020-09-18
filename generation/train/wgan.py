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


def run_train(dataset, generator_class=None, discriminator_class=None, **kwargs):
    def reset_grad():
        generator.zero_grad()
        discriminator.zero_grad()
        
    def data_gen(dataloader):
        while True:
            for signal in dataloader:
                yield signal

    device = 'cpu' if kwargs['cpu'] else 'cuda'   # TODO: (@whiteRa2bit, 2020-09-18) Add to other models
    dataloader = data_gen(DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=True))
    
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

    for epoch in range(kwargs['num_epochs']):
        for _ in range(5):  # TODO: (@whiteRa2bit, 2020-08-30) Replace with kwargs param
            # Sample data
            z = Variable(torch.randn(kwargs['batch_size'], kwargs['latent_dim'])).to(device)
            X = Variable(next(dataloader)).to(device)

            # Dicriminator forward-loss-backward-update
            G_sample = generator(z)
            D_real = discriminator(X)
            D_fake = discriminator(G_sample)

            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_optimizer.step()

            # # Weight clipping
            # for p in discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)  # TODO: (@whiteRa2bit, 2020-08-30) Replace with kwargs param

            # Housekeeping - reset gradient
            reset_grad()

        # Generator forward-loss-backward-update
        X = Variable(next(dataloader)).to(device)
        z = Variable(torch.randn(kwargs['batch_size'], kwargs['latent_dim'])).to(device)

        G_sample = generator(z)
        D_fake = discriminator(G_sample)

        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        G_optimizer.step()

        # Housekeeping - reset gradient
        reset_grad()
        
        if kwargs['verbose'] and epoch % kwargs['print_each'] == 0:
            wandb.log({"D loss": D_loss.cpu().data.numpy(), "G loss": G_loss.cpu().data.numpy()})

            rows_num = 3
            samples = generator(z).cpu().data.numpy()[:rows_num**2]

            f, ax = plt.subplots(rows_num, rows_num, figsize=(rows_num**2, rows_num**2))
            gs = gridspec.GridSpec(rows_num, rows_num)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax[i//rows_num][i % rows_num].plot(sample)
            plt.show()

        kwargs['model_name'] = 'discriminator'
        save_checkpoint(discriminator, epoch, **kwargs)
        kwargs['model_name'] = 'generator'
        save_checkpoint(generator, epoch, **kwargs)

    torch.save(generator.state_dict(), os.path.join(wandb.run.dir, 'generator.pt'))
    torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, 'discriminator.pt'))

    return generator


def generate_new_signal(generator, device='cpu', signals_num=1):   # TODO: (@whiteRa2bit, 2020-08-31) Create shared function for gans
    generator.to(device)
    z = Variable(torch.randn(signals_num + 1, generator.latent_dim)).to(device)
    return generator(z)[:signals_num].cpu().detach().numpy()
