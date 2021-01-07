from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class AbstractGenerator(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config):
        super(AbstractGenerator, self).__init__()
        pass

    @abstractmethod
    def forward(self, x, debug=False):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def visualize(real_sample, generated_sample):
        raise NotImplementedError


class AbstractDiscriminator(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config):
        super(AbstractDiscriminator, self).__init__()
        pass

    @abstractmethod
    def forward(self, x, debug=False):
        raise NotImplementedError
