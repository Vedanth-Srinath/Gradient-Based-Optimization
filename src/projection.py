"""
projection.py
Turns free real-valued logits theta into (near-)binary pixel designs, the way
the pilot chose: continuous relaxation + hard projection, with the tanh
sharpness k ANNEALED low->high over the optimization (unlike the TNN's fixed
k=15, which gives dead gradients for direct logit optimization).

    x = 0.5 * (1 + tanh(k * theta))      # theta=0 -> 0.5 ; theta>0 -> 1 ; <0 -> 0

Small k early: gradients everywhere, pixels move freely.
Large k late : x is essentially binary, so the surrogate sees in-distribution
               (binary) inputs by the end.

Feed pixels: some pixels are fixed metal (the antenna feed) and must NOT be
optimized. We overwrite them to 1 and, because that overwrite is constant, no
gradient flows to their logits.
"""

import numpy as np
import torch


def anneal_k(step, total, k_min=1.0, k_max=15.0):
    """Linear ramp of tanh sharpness from k_min to k_max over `total` steps."""
    if total <= 1:
        return k_max
    frac = min(max(step / (total - 1), 0.0), 1.0)
    return k_min + (k_max - k_min) * frac


def relax(theta, k):
    """Logits -> soft pixels in [0,1]."""
    return 0.5 * (1.0 + torch.tanh(k * theta))


def apply_feed(x, feed_mask):
    """Force feed pixels to 1. feed_mask: bool tensor (H,W) or None."""
    if feed_mask is None:
        return x
    fm = feed_mask.to(x.dtype)
    return x * (1.0 - fm) + fm            # constant -> zero grad at feed pixels


def project(x):
    """Hard binarize for the design that actually gets simulated/scored."""
    return (x > 0.5).to(x.dtype)


def detect_feed_pixels(X, thresh=0.99):
    """Infer fixed feed pixels from data: pixels that are 1 in ~all designs.

    X: numpy array (N,1,12,12) or (N,12,12). Returns a (12,12) bool mask.
    This avoids hard-coding MATLAB (6:7,1) indices that could be transposed;
    verify the detected mask looks like a small feed region before trusting it.
    """
    Xa = np.asarray(X)
    if Xa.ndim == 4:
        Xa = Xa[:, 0]
    frac_on = Xa.mean(axis=0)             # (12,12), fraction of designs where pixel=1
    return frac_on >= thresh


if __name__ == "__main__":
    torch.manual_seed(0)
    theta = torch.zeros(3, 1, 12, 12, requires_grad=True)  # 3 restarts
    # fake feed: two pixels always on
    feed = torch.zeros(12, 12, dtype=torch.bool)
    feed[5, 0] = feed[6, 0] = True

    for step in [0, 20, 39]:
        k = anneal_k(step, 40)
        x = apply_feed(relax(theta, k), feed)
        print(f"step {step:2d}  k={k:4.1f}  x[feed]={x[0,0,5,0].item():.3f}  "
              f"x mid={x[0,0,0,0].item():.3f}")

    # gradient flows to non-feed logits, not to feed logits
    x = apply_feed(relax(theta, 5.0), feed)
    x.sum().backward()
    g = theta.grad[0, 0]
    print("grad at feed (should be 0):", float(g[5, 0]))
    print("grad at non-feed (nonzero):", float(g[0, 0]))
    print("projection unique:", torch.unique(project(x)).tolist())
