import numpy as np


def min_max(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return (x_max - x) / (x_max - x_min)


def standard(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (x - x_mean) / x_std
