"""
data_utils.py
Load the pixelant antenna dataset and provide a single Normalizer that both
training and the design loss share, so the dB<->normalized conversion is
defined in exactly one place and cannot drift.

Confirmed dataset facts (inspected 2026-08-13):
  XTrain1 : (677479, 1, 12, 12) uint8, binary {0,1}   -> already conv-ready
  YTrain  : (81, 677479) float64, S11 in dB           -> TRANSPOSE to (N, 81)
            negative = good (deep resonance); rare values > 0 are sim noise.
"""

from dataclasses import dataclass
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset


# ----------------------------------------------------------------------------- 
# Normalizer: the ONE place dB <-> normalized is defined.
# Both the training loop and loss.py import this. mu/sigma are saved in the
# checkpoint so they are never recomputed (recomputing risks a silent mismatch).
# -----------------------------------------------------------------------------
@dataclass
class Normalizer:
    mu: float
    sigma: float

    def normalize(self, y_db):
        """dB -> normalized space (what the network predicts)."""
        return (y_db - self.mu) / self.sigma

    def to_db(self, y_norm):
        """normalized space -> dB. Differentiable; used inside L_spec.

        Works for both torch tensors and numpy arrays.
        """
        return y_norm * self.sigma + self.mu

    def state(self):
        return {"mu": float(self.mu), "sigma": float(self.sigma)}

    @classmethod
    def from_state(cls, state):
        return cls(mu=state["mu"], sigma=state["sigma"])


@dataclass
class Dataset:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    normalizer: Normalizer


def load_dataset(
    path,
    val_frac=0.1,
    test_frac=0.1,
    seed=0,
    clamp_positive_to=0.0,
):
    """Load, clean, split, and normalize the antenna dataset.

    Returns a Dataset with train/val/test TensorDatasets of
    (X float32 (N,1,12,12), Y_norm float32 (N,81)) plus the Normalizer.
    """
    m = scipy.io.loadmat(path)

    X = m["XTrain1"].astype(np.float32)          # (N,1,12,12), already {0,1}
    Y = m["YTrain"].astype(np.float32).T         # (81,N) -> (N,81)

    # Rare S11 > 0 dB are non-physical simulation noise; clamp so neither the
    # surrogate nor GD can exploit them.
    Y = np.minimum(Y, clamp_positive_to)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    # Normalization stats from TRAIN ONLY (never the full set -> no leakage).
    mu = float(Y[train_idx].mean())
    sigma = float(Y[train_idx].std())
    norm = Normalizer(mu=mu, sigma=sigma)

    def make(idx):
        xb = torch.from_numpy(X[idx])
        yb = torch.from_numpy(norm.normalize(Y[idx]))
        return TensorDataset(xb, yb)

    return Dataset(
        train=make(train_idx),
        val=make(val_idx),
        test=make(test_idx),
        normalizer=norm,
    )


if __name__ == "__main__":
    # quick self-check on synthetic data shaped like the real thing
    fake = {
        "XTrain1": (np.random.rand(1000, 1, 12, 12) > 0.5).astype(np.uint8),
        "YTrain": (np.random.randn(81, 1000) * 4 - 3).astype(np.float64),
    }
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "fake.mat")
    scipy.io.savemat(p, fake)
    ds = load_dataset(p)
    xb, yb = ds.train[0]
    print("train n:", len(ds.train), "x:", tuple(xb.shape), "y:", tuple(yb.shape))
    print("mu,sigma:", round(ds.normalizer.mu, 3), round(ds.normalizer.sigma, 3))
    # round-trip check
    y0 = yb.numpy()
    back = ds.normalizer.to_db(y0)
    fwd = ds.normalizer.normalize(back)
    print("roundtrip max err:", float(np.abs(fwd - y0).max()))
