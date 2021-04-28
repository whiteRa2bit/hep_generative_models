import sys
sys.path.append('/home/pafakanov/hse/hep_generative_models/tools/train')

import torch

from generation.training.wgan_trainer import WganTrainer
from generation.utils import set_seed

from dataset import MyDataset
from model import Generator, Discriminator
from config import CONFIG
from scheduler import get_gpu_id
from generation.utils import set_seed

set_seed()
CONFIG["device"] = f"cuda:{get_gpu_id()}"
dataset = MyDataset()
generator = Generator(CONFIG)
discriminator = Discriminator(CONFIG)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=CONFIG['g_lr'])
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=CONFIG['d_lr'])

trainer = WganTrainer(generator, discriminator, g_optimizer, d_optimizer, CONFIG)
trainer.run_train(dataset)
