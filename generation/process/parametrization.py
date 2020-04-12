import numpy as np
from scipy import signal as signal_processing
import matplotlib.pyplot as plt


class GaussianParametrizer:
    def __init__(self, compress_coef, gaussian_std, model):
        self.compress_coef = compress_coef
        self.gaussian_std = gaussian_std
        self.model = model
        self.gaussians = None

    def intialize_gaussians(self, gaussians):
        self.gaussians = gaussians if not self.gaussians else None

    def get_gaussians(self, signal):
        points_num = 2 * len(signal) + 1
        max_loc = np.argmax(signal_processing.gaussian(points_num, std=self.gaussian_std))

        gaussians = []
        for i in range(0, len(signal), self.compress_coef):
            start_pos = max_loc - i - 1
            end_pos = start_pos + len(signal)
            next_gaussian = signal_processing.gaussian(points_num, std=self.gaussian_std)[start_pos:end_pos]
            gaussians.append(next_gaussian)
        self.intialize_gaussians(gaussians)
        return gaussians

    def transform_signal(self, signal, to_visualize=False):
        def visualize():
            f, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].set_title("Gaussians")
            ax[1].set_title("Weighted gaussians")
            ax[2].set_title("Result")
            result = [0] * len(signal)
            for i, gaussian in enumerate(gaussians):
                ax[0].plot(gaussian)
                ax[1].plot(gaussian * reg.coef_[i])
                result += gaussian * reg.coef_[i]
            ax[2].plot(result)
            ax[2].plot(signal)
            ax[2].legend(['Gaussians sum', 'Origin signal'])
            plt.show()

        reg = self.model.copy()  # Make sure that fit_intercept is set to false
        gaussians = self.get_gaussians(signal)
        cur_X = np.array(gaussians).T
        cur_y = signal.copy()
        reg.fit(cur_X, cur_y)

        if to_visualize:
            visualize()

        return reg.coef_

    def transform_data(self, data):
        new_data = []
        for signal in data:
            coef = self.transform_signal(signal)
            new_data.append(coef)
        return new_data
