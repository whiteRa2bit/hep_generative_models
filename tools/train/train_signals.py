import torch

from generation.config import SIGNALS_TRAINING_CONFIG as CONFIG
from generation.dataset.signals_dataset import SignalsDataset
from generation.nets.signals_net import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer
from generation.utils import set_seed
from scheduler import get_gpu_id


def run_train(config=CONFIG):
    config['device'] = f"cuda:{get_gpu_id()}"
    dataset = SignalsDataset(signal_size=config['x_dim'], freq=config["x_freq"])

    generator = Generator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config['g_lr'], betas=(config["g_beta1"], config["g_beta2"]))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config['d_lr'], betas=(config["d_beta1"], config["d_beta2"]))

    trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, config)
    trainer.run_train(dataset)


if __name__ == '__main__':
    set_seed()
    run_train()
