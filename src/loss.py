"""
loss.py
Inverse-design loss. This scores a candidate design against a SPEC (not a target
curve) and is backpropagated to the pixel logits.

    L = L_spec  +  beta * L_bin  [+ gamma * L_trust]   (L_trust added in optimize)

L_spec uses a one-sided SOFTPLUS hinge, not MSE:
  * passband: want S11 deep (<= target_db). Penalize only when too shallow.
  * stopband: want S11 high (>= stop_floor_db). Penalize only when too deep.
Softplus (not ReLU) keeps a small gradient once a band is satisfied, so the
optimizer doesn't stall exactly at the threshold. There is no unique target
CURVE -- only the spec -- so MSE would wrongly penalize designs that beat spec.

Frequency grid: 81 points over 10-20 GHz => 125 MHz per bin.
"""

import numpy as np
import torch
import torch.nn.functional as F


N_FREQ = 81
F_LO_GHZ, F_HI_GHZ = 10.0, 20.0


def bin_to_ghz(idx):
    return F_LO_GHZ + idx * (F_HI_GHZ - F_LO_GHZ) / (N_FREQ - 1)


def ghz_to_bin(ghz):
    return int(round((ghz - F_LO_GHZ) * (N_FREQ - 1) / (F_HI_GHZ - F_LO_GHZ)))


def make_masks(centers, half_pass=4, guard=1, n_freq=N_FREQ):
    """Build pass/stop boolean masks over the frequency grid.

    centers  : list of center bin indices (1 entry = single-band, 2 = dual-band)
    half_pass: passband half-width in bins (±half_pass)
    guard    : transition bins on each side of each passband, excluded from BOTH
               pass and stop (a hard pass/stop edge is physically unrealisable
               and creates a permanent gradient tug-of-war).
    Returns (pass_mask, stop_mask) as bool numpy arrays of length n_freq.
    """
    pass_mask = np.zeros(n_freq, dtype=bool)
    guard_mask = np.zeros(n_freq, dtype=bool)
    for c in centers:
        lo, hi = c - half_pass, c + half_pass
        pass_mask[max(0, lo): hi + 1] = True
        g_lo = max(0, lo - guard)
        g_hi = min(n_freq - 1, hi + guard)
        guard_mask[g_lo: g_hi + 1] = True
    stop_mask = (~guard_mask)             # everything not pass-or-guard
    return pass_mask, stop_mask


def spec_loss(
    s11_db,
    pass_mask,
    stop_mask,
    target_db=-8.0,
    stop_floor_db=-2.0,
    w_pass=1.0,
    w_stop=1.0,
    db_floor=-40.0,
):
    """Softplus-hinge spec loss.

    s11_db   : tensor (R, n_freq) predicted S11 in dB (R = restarts/batch)
    *_mask   : bool tensors (n_freq,)
    Returns  : (loss_total, dict of components) ; loss_total shape (R,)
    """
    s = s11_db.clamp(min=db_floor)        # don't chase depths the surrogate can't vouch for
    pm = pass_mask.to(s.device)
    sm = stop_mask.to(s.device)

    # passband: penalize s > target (too shallow)
    pass_pen = F.softplus(s[:, pm] - target_db).mean(dim=1)
    # stopband: penalize s < floor (too deep)
    stop_pen = F.softplus(stop_floor_db - s[:, sm]).mean(dim=1)

    total = w_pass * pass_pen + w_stop * stop_pen
    return total, {"pass": pass_pen.detach(), "stop": stop_pen.detach()}


def binary_penalty(x):
    """0 when x is binary, 1 at x=0.5. Mean over all pixels, per restart."""
    return (4.0 * x * (1.0 - x)).mean(dim=(1, 2, 3))


if __name__ == "__main__":
    torch.manual_seed(0)
    c = ghz_to_bin(15.0)                  # center at 15 GHz
    pm, sm = make_masks([c], half_pass=4, guard=1)
    print(f"center bin {c} = {bin_to_ghz(c):.2f} GHz")
    print(f"pass bins: {int(pm.sum())}, stop bins: {int(sm.sum())}, "
          f"guard bins: {N_FREQ - int(pm.sum()) - int(sm.sum())}")

    pm_t = torch.from_numpy(pm)
    sm_t = torch.from_numpy(sm)
    # fake: a good design (deep in passband) vs a flat design
    good = torch.full((1, N_FREQ), -1.0)
    good[0, pm] = -12.0
    flat = torch.full((1, N_FREQ), -1.0)
    lg, _ = spec_loss(good, pm_t, sm_t, target_db=-8.0)
    lf, _ = spec_loss(flat, pm_t, sm_t, target_db=-8.0)
    print(f"loss good={lg.item():.4f}  loss flat={lf.item():.4f}  "
          f"(good should be lower)")

    x = torch.rand(2, 1, 12, 12)
    print("binary_penalty (random ~0.33):", binary_penalty(x).tolist())
