import numpy as np
from utils import probability


class GaussianNBClassifier:
    def __init__(
        self, features: np.ndarray, classes: np.ndarray, eps: float = 1e-4
    ) -> None:
        mean, cov = [], []
        cls_unique, cls_freq = np.unique(classes, return_counts=True)
        for label in cls_unique:
            feat = features[classes == label]
            mean.append(np.mean(feat, 0))
            cov.append(np.cov(feat.T, ddof=1))
        self.mean = np.stack(mean)
        self.cov = np.stack(cov)
        # Inverse covariance matrix = precision matrix
        self.prec = np.linalg.inv(self.cov)
        self.det = np.linalg.det(self.cov)
        self.prior = cls_freq / cls_freq.sum()
        self.eps = eps

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Fetch feature shape
        d = self.cov.shape[-1]
        # Calculate P(y|X=x) ~ N(z, mu, sigma)
        # Calculate z - mu
        y = inputs - self.mean[:, None]
        # Batch matrix multiplication
        # b -> batch
        # c -> class
        # e-> feature dimension (row-wise)
        # d -> features dimension (col-wise)
        p = np.einsum("cbd,ced->bcd", y, self.prec)
        # Dot product on features
        p = np.einsum("bcd,cbd->bc", p, y)
        p = np.exp(-p / 2)
        p = p / self.det**0.5 * (2 * np.pi) ** (d / 2)
        # Calculate P(y|X=x) * P(y)
        p = p * self.prior
        # Return probability
        return probability(p, eps=self.eps)
