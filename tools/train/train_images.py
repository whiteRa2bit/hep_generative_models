import torch

from generation.config import IMAGES_TRAINING_CONFIG as CONFIG
from generation.dataset.images_dataset import ImageDataset
from generation.nets.images import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer
from generation.training.utils import set_seed


def run_train(config=CONFIG):
    dataset = ImageDataset(config['detector'])

    generator = Generator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['lr'])

    trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, config)

    trainer.run_train(dataset)


if __name__ == '__main__':
    set_seed()
    run_train()
