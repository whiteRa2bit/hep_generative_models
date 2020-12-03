import os

import torch
from torch.autograd import Variable
from loguru import logger

from generation.utils import get_config_path, read_json, get_checkpoint_dir


class InferenceModel:
    def __init__(self, generator_class, run_id, epoch=None, batch_size=None):
        self.run_id = run_id
        self.config = read_json(get_config_path(run_id))
        self.batch_size = batch_size or self.config["batch_size"]
        self.generator = generator_class(self.config).to(self.config["device"])
        self._load_checkpoint(epoch)

    def _load_checkpoint(self, epoch=None):
        last_epoch = self.config["save_each"] * ((self.config["epochs_num"] - 1) // self.config["save_each"])
        epoch = epoch or last_epoch
        checkpoint_dir = get_checkpoint_dir(self.run_id)
        checkpoint_path = os.path.join(checkpoint_dir, f"generator_{epoch}.pt")
        self.generator.load_state_dict(torch.load(checkpoint_path))
        self.generator.eval()
        logger.info(f"Restored checkpoint from epoch {epoch}")

    @torch.no_grad()
    def generate(self, samples_num=1):
        samples = []
        for i in range((samples_num - 1) // self.batch_size + 1):
            if i == samples_num // self.batch_size:
                batch_size = samples_num % self.batch_size
            else:
                batch_size = self.batch_size
            z = Variable(torch.randn(batch_size, self.config['z_dim']))
            z = z.to(self.config['device'])
            sample = self.generator(z)
            samples.append(sample)
        samples = torch.cat(samples)
        return samples
