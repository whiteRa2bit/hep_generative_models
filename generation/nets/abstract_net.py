from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class AbstractGenerator(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def forward(self, x, debug=False):
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def visualize(generated_sample, real_sample):
        raise NotImplementedError


class AbstractDiscriminator(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def forward(self, x, debug=False):
        raise NotImplementedError
