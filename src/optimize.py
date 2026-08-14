"""
optimize.py
Gradient-based inverse design: optimize pixel logits through the FROZEN forward
surrogate to meet an S11 spec.

Pipeline per target:
  theta (R restarts) --relax(k)--> soft pixels --apply_feed--> x
      --surrogate--> S11(norm) --to_db--> S11(dB) --spec_loss--> L
  L = L_spec + beta*L_bin + gamma*L_trust ; Adam on theta ; anneal k, beta.
At the end we hard-PROJECT and report the score BOTH on the soft design and the
projected binary design -- the gap is a key diagnostic.

Run:
    python src/optimize.py --ckpt models/seed0_best.pth --center_ghz 15 --target_db -8
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from data_utils import Normalizer                      # noqa: E402
from model import build_model                          # noqa: E402
from projection import anneal_k, relax, apply_feed, project  # noqa: E402
from loss import spec_loss, binary_penalty, make_masks, ghz_to_bin, bin_to_ghz  # noqa: E402


def load_surrogate(ckpt_path, device, dropout=0.1):
    """Rebuild model + normalizer from a training checkpoint."""
    ck = torch.load(ckpt_path, map_location=device)
    model = build_model(p_drop=dropout).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = Normalizer(mu=ck["mu"], sigma=ck["sigma"])
    return model, norm


def enable_mc_dropout(model):
    """Put ONLY dropout layers in train mode (keep GroupNorm in eval)."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def trust_variance(model, x, norm, K=8):
    """MC-dropout predictive variance in dB, mean over freq. Higher = less trust.

    Uses one checkpoint; with a 3-seed ensemble this would be cross-seed variance.
    """
    enable_mc_dropout(model)
    preds = []
    for _ in range(K):
        preds.append(norm.to_db(model(x)))
    model.eval()
    stack = torch.stack(preds, dim=0)                  # (K, R, 81)
    return stack.var(dim=0).mean(dim=1)                # (R,)


def optimize_target(
    model, norm, centers, device,
    target_db=-8.0, stop_floor_db=-2.0,
    half_pass=4, guard=1,
    restarts=64, iters=300, lr=0.1,
    k_min=1.0, k_max=15.0,
    beta_max=2.0, gamma=0.0, trust_K=8,
    feed_mask=None, seed=0,
):
    torch.manual_seed(seed)
    pm_np, sm_np = make_masks(centers, half_pass=half_pass, guard=guard)
    pm = torch.from_numpy(pm_np).to(device)
    sm = torch.from_numpy(sm_np).to(device)
    if feed_mask is not None:
        feed_mask = feed_mask.to(device)

    theta = (0.1 * torch.randn(restarts, 1, 12, 12, device=device)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)

    for step in range(iters):
        k = anneal_k(step, iters, k_min, k_max)
        beta = beta_max * (step / max(iters - 1, 1))   # ramp binary penalty
        x = apply_feed(relax(theta, k), feed_mask)
        s11_db = norm.to_db(model(x))
        l_spec, _ = spec_loss(s11_db, pm, sm, target_db, stop_floor_db)
        l_bin = binary_penalty(x)
        loss = l_spec + beta * l_bin
        if gamma > 0:
            loss = loss + gamma * trust_variance(model, x, norm, K=trust_K)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()

    # ---- evaluate: soft vs projected ----
    with torch.no_grad():
        x_soft = apply_feed(relax(theta, k_max), feed_mask)
        x_proj = apply_feed(project(x_soft), feed_mask)

        soft_db = norm.to_db(model(x_soft))
        proj_db = norm.to_db(model(x_proj))
        soft_score, _ = spec_loss(soft_db, pm, sm, target_db, stop_floor_db)
        proj_score, _ = spec_loss(proj_db, pm, sm, target_db, stop_floor_db)

        best = int(torch.argmin(proj_score))
        result = {
            "best_design": x_proj[best, 0].cpu().numpy().astype(np.uint8),
            "best_proj_score": float(proj_score[best]),
            "best_soft_score": float(soft_score[best]),
            "best_s11_db": proj_db[best].cpu().numpy(),
            "proj_score_all": proj_score.cpu().numpy(),
            "soft_score_all": soft_score.cpu().numpy(),
            "projection_gap": float((proj_score - soft_score).mean()),
            "pass_mask": pm_np, "stop_mask": sm_np,
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed0_best.pth")
    p.add_argument("--center_ghz", type=float, nargs="+", default=[15.0])
    p.add_argument("--target_db", type=float, default=-8.0)
    p.add_argument("--stop_floor_db", type=float, default=-2.0)
    p.add_argument("--restarts", type=int, default=64)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.0, help="L_trust weight (0=off)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    model, norm = load_surrogate(args.ckpt, device)
    centers = [ghz_to_bin(g) for g in args.center_ghz]
    print(f"device={device}  centers(GHz)={[round(bin_to_ghz(c),2) for c in centers]}")

    r = optimize_target(model, norm, centers, device,
                        target_db=args.target_db, stop_floor_db=args.stop_floor_db,
                        restarts=args.restarts, iters=args.iters, gamma=args.gamma)
    passband_db = r["best_s11_db"][r["pass_mask"]]
    print(f"best projected spec-loss : {r['best_proj_score']:.4f}")
    print(f"best soft spec-loss      : {r['best_soft_score']:.4f}")
    print(f"projection gap (mean)    : {r['projection_gap']:.4f}")
    print(f"passband S11 (dB) min/max: {passband_db.min():.2f} / {passband_db.max():.2f}")
    print(f"target was <= {args.target_db} dB in passband")


if __name__ == "__main__":
    main()
