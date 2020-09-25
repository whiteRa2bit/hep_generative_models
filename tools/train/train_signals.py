import torch

from generation.config import SIGNALS_TRAINING_CONFIG as CONFIG
from generation.dataset.signals_dataset import SignalsDataset
from generation.nets.shapes_1d import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer


def run_train(config=CONFIG):
    dataset = SignalsDataset(config['detector'], signal_size=config['x_dim'])

    generator = Generator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['lr'])

    trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, config)

    trainer.run_train(dataset)


if __name__ == '__main__':
    run_train()
