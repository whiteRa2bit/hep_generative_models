import numpy as np
from scipy.stats import nakagami
import matplotlib.pyplot as plt


class Nakagami:
    def __init__(self, sample_size, q_lower, q_upper):
        self.sample_size = sample_size
        self.q_lower = q_lower
        self.q_upper = q_upper

    def _get_nakagami_sample(self, nu):
        x = np.linspace(nakagami.ppf(self.q_lower, nu),
                        nakagami.ppf(self.q_upper, nu),
                        self.sample_size)

        return nakagami.pdf(x, nu)

    def get_nakagami_data(self, nu_values):
        dataset = []
        for nu_value in nu_values:
            sample = self._get_nakagami_sample(nu_value)
            sample /= np.max(sample)
            dataset.append(sample)
        return np.array(dataset)

    @staticmethod
    def visualize_nakagami_dataset(dataset):
        for sample in dataset:
            plt.plot(sample)
        plt.show()
