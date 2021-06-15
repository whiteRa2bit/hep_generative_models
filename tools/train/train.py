import argparse

import torch

from generation.config import AMPLITUDES_MODEL_NAME, IMAGES_MODEL_NAME, SHAPES_MODEL_NAME, SIGNALS_MODEL_NAME, \
    SIMPLIFIED_MODEL_NAME, AMPLITUDES_TRAINING_CONFIG, IMAGES_TRAINING_CONFIG, SHAPES_TRAINING_CONFIG, \
    SIGNALS_TRAINING_CONFIG, SIMPLIFIED_MODEL_CONFIG
from generation.dataset.amplitudes_dataset import AmplitudesDataset
from generation.nets.amplitudes_net import AmplitudesGenerator, AmplitudesDiscriminator
from generation.dataset.images_dataset import ImagesDataset
from generation.nets.images_net import ImagesGenerator, ImagesDiscriminator
from generation.dataset.shapes_dataset import ShapesDataset
from generation.nets.shapes_net import ShapesGenerator, ShapesDiscriminator
from generation.dataset.signals_dataset import SignalsDataset
from generation.nets.signals_net import SignalsGenerator, SignalsDiscriminator
from generation.dataset.simplified_dataset import SimplifiedDataset
from generation.nets.simplified_net import SimplifiedGenerator, SimplifiedDiscriminator
from generation.training.gan_trainer import GanTrainer
from generation.training.wgan_trainer import WganTrainer
from generation.utils import set_seed
from scheduler import get_gpu_id


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-m',
        '--model_name',
        choices=[
            AMPLITUDES_MODEL_NAME, IMAGES_MODEL_NAME, SHAPES_MODEL_NAME, SIGNALS_MODEL_NAME, SIMPLIFIED_MODEL_NAME
        ],
        required=True)
    args = argparser.parse_args()
    return args


def _get_params_by_model_name(model_name):
    if model_name == AMPLITUDES_MODEL_NAME:
        config = AMPLITUDES_TRAINING_CONFIG
        dataset = AmplitudesDataset()
        generator = AmplitudesGenerator(config)
        discriminator = AmplitudesDiscriminator(config)
        trainer_class = GanTrainer
    elif model_name == IMAGES_MODEL_NAME:
        config = IMAGES_TRAINING_CONFIG
        dataset = ImagesDataset(detector=config["detector"])
        generator = ImagesGenerator(config)
        discriminator = ImagesDiscriminator(config)
        trainer_class = WganTrainer
    elif model_name == SHAPES_MODEL_NAME:
        config = SHAPES_TRAINING_CONFIG
        dataset = ShapesDataset(detector=config["detector"], signal_dim=config["x_dim"])
        generator = ShapesGenerator(config)
        discriminator = ShapesDiscriminator(config)
        trainer_class = WganTrainer
    elif model_name == SIGNALS_MODEL_NAME:
        config = SIGNALS_TRAINING_CONFIG
        dataset = SignalsDataset(signal_dim=config["x_dim"], freq=config["x_freq"])
        generator = SignalsGenerator(config)
        discriminator = SignalsDiscriminator(config)
        trainer_class = WganTrainer
    elif model_name == SIMPLIFIED_MODEL_NAME:
        config = SIMPLIFIED_MODEL_CONFIG
        dataset = SimplifiedDataset(signal_dim=config["x_dim"], freq=config["x_freq"])
        generator = SimplifiedGenerator(config)
        discriminator = SimplifiedDiscriminator(config)
        trainer_class = WganTrainer

    return dataset, generator, discriminator, trainer_class, config


def run_train(model_name):
    dataset, generator, discriminator, trainer_class, config = _get_params_by_model_name(model_name)
    config["device"] = f"cuda:{get_gpu_id()}"

    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=config['g_lr'], betas=(config["g_beta1"], config["g_beta2"]))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=config['d_lr'], betas=(config["d_beta1"], config["d_beta2"]))
    trainer = trainer_class(generator, discriminator, g_optimizer, d_optimizer, config)

    trainer.run_train(dataset)


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    run_train(args.model_name)
