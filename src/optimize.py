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
# build_model is imported inside load_surrogate based on ck["arch"]
from projection import anneal_k, relax, apply_feed, project  # noqa: E402
from loss import spec_loss, binary_penalty, make_masks, ghz_to_bin, bin_to_ghz  # noqa: E402


def load_surrogate(ckpt_path, device, dropout=0.1):
    """Rebuild model + normalizer from a training checkpoint.

    Dispatches on ck["arch"] so old CNN checkpoints (which have no such key)
    still load unchanged; they default to arch='cnn'. Transformer checkpoints
    saved by train_surrogate.py --arch transformer carry arch='transformer'.
    """
    ck = torch.load(ckpt_path, map_location=device)
    arch = ck.get("arch", "cnn")
    if arch == "transformer":
        from model_transformer import build_model
    else:
        from model import build_model
    model = build_model(p_drop=dropout).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = Normalizer(mu=ck["mu"], sigma=ck["sigma"])
    print(f"loaded arch={arch} from {ckpt_path}")
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
    feed_mask=None, seed=0, warm_init=None, warm_frac=0.5,
):
    torch.manual_seed(seed)
    pm_np, sm_np = make_masks(centers, half_pass=half_pass, guard=guard)
    pm = torch.from_numpy(pm_np).to(device)
    sm = torch.from_numpy(sm_np).to(device)
    if feed_mask is not None:
        feed_mask = feed_mask.to(device)

    # Initialize logits. Random by default; if warm_init designs are given,
    # seed a fraction of restarts FROM those real designs (mapped to logits)
    # so GD refines shapes the surrogate already predicts well, instead of
    # inventing out-of-distribution geometry. warm_init: (M,12,12) binary array.
    theta = 0.1 * torch.randn(restarts, 1, 12, 12, device=device)
    if warm_init is not None and len(warm_init) > 0:
        n_warm = int(round(restarts * warm_frac))
        wi = torch.as_tensor(warm_init, dtype=torch.float32, device=device)  # (M,12,12)
        idx = torch.randint(0, wi.shape[0], (n_warm,), device=device)
        seed_designs = wi[idx]                                  # (n_warm,12,12)
        # binary {0,1} -> logits: 1 -> +3, 0 -> -3 (tanh(k*3) ~ binary at low k),
        # plus small noise so the n_warm restarts aren't identical.
        seed_logits = (seed_designs * 2.0 - 1.0) * 3.0
        seed_logits = seed_logits.unsqueeze(1)                  # (n_warm,1,12,12)
        seed_logits = seed_logits + 0.3 * torch.randn_like(seed_logits)
        theta[:n_warm] = seed_logits
    theta = theta.requires_grad_(True)
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


def save_result_mat(path, design_12x12, s11_db_81, target_f_ghz, target_d_db):
    """Write ONE design in the exact format the TNN MATLAB validator reads:
       pred_antenna_rec (12x12), pred_spec_rec (81x1 dB),
       target_freq_GHz, target_depth_dB.
    """
    import scipy.io
    scipy.io.savemat(path, {
        "pred_antenna_rec": design_12x12.astype(np.float64),
        "pred_spec_rec": np.asarray(s11_db_81, dtype=np.float64).reshape(-1, 1),
        "target_freq_GHz": float(target_f_ghz),
        "target_depth_dB": float(target_d_db),
    })


def load_warm_designs(data_path, center_bin, device, tol_bins=3, depth_max=-6.0, cap=200):
    """Find real dataset designs whose S11 dips near `center_bin`.

    Returns up to `cap` binary (M,12,12) designs whose minimum S11 in the window
    [center_bin +/- tol_bins] is below depth_max (dB). These are in-distribution
    shapes that already resonate near the target -> good GD starting points.
    """
    import scipy.io
    d = scipy.io.loadmat(data_path)
    X = d["XTrain1"]                      # (N,1,12,12) uint8
    Y = d["YTrain"]                       # (81,N) dB
    lo = max(0, center_bin - tol_bins)
    hi = min(Y.shape[0] - 1, center_bin + tol_bins)
    window_min = Y[lo:hi + 1, :].min(axis=0)          # (N,) deepest dip near target
    sel = np.where(window_min < depth_max)[0]
    if len(sel) == 0:
        print(f"  warm-start: no dataset design dips < {depth_max} dB near bin "
              f"{center_bin}; falling back to random init.")
        return None
    if len(sel) > cap:
        sel = np.random.default_rng(0).choice(sel, cap, replace=False)
    designs = X[sel, 0].astype(np.float32)            # (M,12,12)
    print(f"  warm-start: {len(sel)} dataset designs resonate near bin {center_bin}")
    return designs


def detect_feed_from_data(data_path, device, thresh=0.99, max_n=20000):
    """Load a chunk of the dataset and infer fixed feed pixels."""
    import scipy.io
    from projection import detect_feed_pixels
    X = scipy.io.loadmat(data_path)["XTrain1"][:max_n]
    mask_np = detect_feed_pixels(X, thresh=thresh)
    n = int(mask_np.sum())
    print(f"feed pixels detected: {n} at {list(zip(*np.where(mask_np)))}")
    if n == 0 or n > 8:
        print("  WARNING: feed detection looks off (expected ~1-4 pixels). "
              "Check orientation / thresh before trusting designs.")
    return torch.from_numpy(mask_np).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed0_best.pth")
    p.add_argument("--data", default=None,
                   help="dataset path; if given, feed pixels are detected and fixed")
    p.add_argument("--out_dir", default="results/gd_designs")
    p.add_argument("--freqs", type=float, nargs="+", default=[15.0])
    p.add_argument("--depths", type=float, nargs="+", default=[-8.0])
    p.add_argument("--stop_floor_db", type=float, default=-2.0)
    p.add_argument("--half_pass", type=int, default=3)
    p.add_argument("--restarts", type=int, default=128)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--gamma", type=float, default=0.0, help="L_trust weight (0=off)")
    p.add_argument("--warm_start", action="store_true",
                   help="seed a fraction of restarts from real dataset designs "
                        "that resonate near the target (needs --data)")
    p.add_argument("--warm_frac", type=float, default=0.5,
                   help="fraction of restarts to warm-start (rest stay random)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    model, norm = load_surrogate(args.ckpt, device)
    print(f"device={device}")

    feed_mask = None
    if args.data:
        feed_mask = detect_feed_from_data(args.data, device)
    else:
        print("WARNING: no --data given -> feed pixels NOT fixed. Designs may be "
              "out-of-distribution for the surrogate and invalid antennas. "
              "Pass --data for valid results.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'target':>16}  {'surr_dip_dB':>11}  {'proj_gap':>8}  file")
    summary = []
    for f_ghz in args.freqs:
        for d_db in args.depths:
            centers = [ghz_to_bin(f_ghz)]
            warm = None
            if args.warm_start and args.data:
                warm = load_warm_designs(args.data, centers[0], device,
                                         depth_max=max(d_db, -6.0))
            r = optimize_target(
                model, norm, centers, device,
                target_db=d_db, stop_floor_db=args.stop_floor_db,
                half_pass=args.half_pass, restarts=args.restarts,
                iters=args.iters, gamma=args.gamma, feed_mask=feed_mask,
                warm_init=warm, warm_frac=args.warm_frac,
            )
            pm = r["pass_mask"]
            surr_dip = float(r["best_s11_db"][pm].min())
            fname = f"pred_rec_f{int(f_ghz)}_d{int(d_db)}.mat"
            fpath = os.path.join(args.out_dir, fname)
            save_result_mat(fpath, r["best_design"], r["best_s11_db"], f_ghz, d_db)
            print(f"{f_ghz:6.0f}GHz/{d_db:5.0f}dB  {surr_dip:11.2f}  "
                  f"{r['projection_gap']:8.3f}  {fname}")
            summary.append((f_ghz, d_db, surr_dip, r["projection_gap"]))

    print(f"\nSaved {len(summary)} designs to {args.out_dir}")
    print("Next: download that folder to your Mac, point baseline_dir at it, "
          "run your TNN MATLAB validator to get EM-vs-surrogate plots.")


if __name__ == "__main__":
    main()