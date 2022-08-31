import numpy as np
from .functional import *


class MinMaxScaler:
    def __init__(self, features: np.ndarray) -> None:
        self.max = np.max(features, 0)
        self.min = np.min(features, 0)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return min_max(inputs, self.min, self.max)


class StandardScaler:
    def __init__(self, features: np.ndarray) -> None:
        self.mean = np.mean(features, 0)
        self.std = np.std(features, 0)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return standard(inputs, self.mean, self.std)
