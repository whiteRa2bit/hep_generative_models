import torch

from generation.config import SIGNALS_TRAINING_CONFIG as CONFIG
from generation.dataset.signals_dataset import SignalsDataset
from generation.nets.signals import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer
from generation.training.utils import set_seed


def main(config=CONFIG):
    z = torch.randn((config["batch_size"], config["z_dim"]))
    generator = Generator(config)
    discriminator = Discriminator(config)
    preds = discriminator(generator(z, debug=True), debug=True)
    print(generator)
    print(discriminator)

if __name__ == '__main__':
    main()
