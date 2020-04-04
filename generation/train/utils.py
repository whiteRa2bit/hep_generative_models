import torch
import argparse
import os
from generation.config import (
    CHECKPOINTS_DIR
)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, help='Size of single signal')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension of generator')
    parser.add_argument('--print_each', type=int, default=20, help='number of epochs between print messages')
    parser.add_argument('--verbose', action="store_false", help='increase output verbosity')
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    parser.add_argument('--save_dir', type=str, help='dir to save model checkpoints', default='')
    parser.add_argument('--save_interval', type=int, default=10, help='save a checkpoint every N epochs')
    parser.add_argument('--no_save', action='store_true', help='don’t save models or checkpoints')
    parser.add_argument('--model_name', type=str, default='gan', help='model name while saving')
    return parser.parse_args()
