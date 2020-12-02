import torch

from generation.config import SIGNALS_TRAINING_CONFIG as CONFIG
from generation.dataset.signals_dataset import SignalsDataset
from generation.nets.signals_net import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer
from generation.utils import set_seed
from scheduler import get_gpu_id


def run_train(config=CONFIG):
    config['device'] = f"cuda:{get_gpu_id()}"
    dataset = SignalsDataset(signal_size=config['x_dim'])

    generator = Generator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['g_lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['d_lr'])

    trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, config)
    trainer.run_train(dataset)


if __name__ == '__main__':
    set_seed()
    run_train()
