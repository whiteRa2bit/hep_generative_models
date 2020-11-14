import torch

from generation.config import AMPLITUDES_TRAINING_CONFIG as CONFIG
from generation.dataset.amplitudes_dataset import AmplitudesDataset
from generation.nets.amplitudes import Generator, Discriminator
from generation.training.wgan_trainer import WganTrainer
from generation.utils import set_seed
from scheduler import get_gpu_id


def run_train(config=CONFIG):
    config['device'] = f"cuda:{get_gpu_id()}"
    dataset = AmplitudesDataset()

    generator = Generator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['lr'])

    trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, config)
    trainer.run_train(dataset)


if __name__ == '__main__':
    set_seed()
    run_train()
