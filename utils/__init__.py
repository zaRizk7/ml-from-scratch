import numpy as np


def probability(y, use_softmax=False, eps: float = 0):
    if use_softmax:
        y = np.exp(y)
    return (y + eps) / (np.sum(y, -1) + eps)[:, None]
