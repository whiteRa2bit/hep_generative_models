from abc import abstractmethod, ABC
import os
import random

import numpy as np
import torch
import wandb

from generation.config import WANDB_PROJECT, CHECKPOINT_DIR, RANDOM_SEED


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

    def _reset_grad(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    @staticmethod
    def _save_checkpoint(model, checkpoint_name):
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, wandb.run.id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)

    @abstractmethod
    def run_train(self, dataset):
        pass
