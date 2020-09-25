import os
from abc import abstractmethod, ABC

import wandb

from generation.config import WANDB_PROJECT


class AbstractTrainer(ABC):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, config):
        self.generator = generator.to(config['device'])
        self.discriminator = discriminator.to(config['device'])
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.config = config

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.generator)
        wandb.watch(self.discriminator)

    def reset_grad(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    @abstractmethod
    def run_train(self, dataset):
        pass
