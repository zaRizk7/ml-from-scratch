import numpy as np
from utils import probability as p


def distance(x1: np.ndarray, x2: np.ndarray, p_norm: int = 2):
    return np.linalg.norm(x1 - x2, p_norm, -1)


def top_k_class(d: np.ndarray, y: np.ndarray, k: int):
    return y[np.argpartition(d, k)[:, :k]]


def probability(y: np.ndarray, num_classes: int, eps: float = 0):
    frequency = lambda y: np.bincount(y, minlength=num_classes)
    y = np.apply_along_axis(frequency, -1, y)
    return p(y, eps=eps)
