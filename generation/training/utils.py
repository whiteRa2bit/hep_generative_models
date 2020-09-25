import argparse
import os

import torch

from generation.config import CHECKPOINTS_DIR


def save_checkpoint(model, epoch, **kwargs):
    if not kwargs['no_save'] and epoch % kwargs['save_interval'] == 0:
        if kwargs['save_dir']:
            save_dir = os.path.join(kwargs['save_dir'], kwargs['model_name'])
        else:
            save_dir = os.path.join(CHECKPOINTS_DIR, kwargs['model_name'])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint_name = os.path.join(save_dir, 'checkpoint_{}.pth'.format(epoch // kwargs['save_interval']))
        torch.save(model.state_dict(), checkpoint_name)
