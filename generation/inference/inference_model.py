import os

import torch
from torch.autograd import Variable

from generation.utils import get_config_path, read_json, get_checkpoint_dir


class InferenceModel:
    def __init__(self, generator_class, run_id, epoch=-1):
        self.run_id = run_id
        self.config = read_json(get_config_path(run_id))
        self.generator = generator_class(self.config).to(self.config["device"])
        self._load_checkpoint(epoch)

    def _load_checkpoint(self, epoch):
        checkpoint_dir = get_checkpoint_dir(self.run_id)
        checkpoint_path = os.path.join(checkpoint_dir, f"generator_{epoch}.pt")
        self.generator.load_state_dict(checkpoint_path)
        self.generator.eval()

    @torch.no_grad()
    def generate(self):
        z = Variable(torch.randn(1, self.config['z_dim']))
        z = z.to(self.config['device'])
        sample = self.generator(z)
        return sample
