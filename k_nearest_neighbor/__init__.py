import numpy as np
from .functional import *


class KNNClassifier:
    def __init__(
        self,
        features: np.ndarray,
        classes: np.ndarray,
        k: int = 1,
        p_norm: int = 2,
        eps: float = 1e-4,
    ) -> None:
        self.features = features
        self.classes = classes
        self.k = k
        self.p_norm = p_norm
        self.eps = eps
        self.num_classes = len(np.unique(classes))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Calculate distance
        d = distance(inputs[:, None], self.features, self.p_norm)
        # Find k-Nearest Neighbor by classes
        y = top_k_class(d, self.classes, self.k)
        # Return probability
        return probability(y, self.num_classes, self.eps)
