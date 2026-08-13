"""
train_surrogate.py
Train the forward CNN surrogate. Runs anywhere:
    python src/train_surrogate.py                 # local (CPU/MPS) or GPU box
    !python src/train_surrogate.py --epochs 40    # Colab
    python src/train_surrogate.py --seed 1        # DSMLP, one of K seeds

Portability:
  * device auto-detects: CUDA -> MPS (Apple) -> CPU.
  * AMP (mixed precision) is enabled ONLY on CUDA; no-ops elsewhere.
  * checkpoints every epoch to models/seed{seed}_last.pth and resumes from it,
    so a Colab/DSMLP disconnect continues instead of restarting.
  * mu/sigma are saved INSIDE the checkpoint (single source of truth).

Gate D (printed at the end): held-out test error in dB, overall and in the
deep-resonance region (true S11 < -10 dB) where the pilot targets live. If the
surrogate is inaccurate there, optimizing through it is not interpretable --
fix before proceeding.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.dirname(__file__))
from data_utils import load_dataset          # noqa: E402
from model import build_model                # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    use_amp = device.type == "cuda"
    print(f"device={device}  amp={use_amp}  seed={args.seed}")

    ds = load_dataset(args.data, seed=args.split_seed)
    norm = ds.normalizer
    print(f"train/val/test = {len(ds.train)}/{len(ds.val)}/{len(ds.test)}  "
          f"mu={norm.mu:.3f} sigma={norm.sigma:.3f}")

    pin = device.type == "cuda"
    train_loader = DataLoader(ds.train, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(ds.val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = build_model(p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Deep-weighted MSE: upweight the rare deep-resonance points (true dB < thr)
    # so the surrogate is accurate where the GD targets live. With
    # --deep_weight 0 this is identical to plain MSE. Weights use the TRUE dB
    # (via the normalizer), so the threshold is in real dB, not normalized units.
    def weighted_mse(pred, y):
        if args.deep_weight <= 0:
            return ((pred - y) ** 2).mean()
        y_db = norm.to_db(y)                       # differentiable, dB space
        w = 1.0 + args.deep_weight * (y_db < args.deep_thr).float()
        return (w * (pred - y) ** 2).mean()

    loss_fn = weighted_mse

    os.makedirs(args.ckpt_dir, exist_ok=True)
    last_path = os.path.join(args.ckpt_dir, f"seed{args.seed}_last.pth")
    best_path = os.path.join(args.ckpt_dir, f"seed{args.seed}_best.pth")

    start_epoch, best_val = 0, float("inf")
    if os.path.exists(last_path):                     # resume
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from epoch {start_epoch}")

    autocast_dev = "cuda" if use_amp else "cpu"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_dev, enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * xb.size(0)
        sched.step()
        train_mse = running / len(ds.train)

        # validation
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                # validation stays PLAIN MSE so val is comparable across
                # different --deep_weight settings (best.pth selection is fair).
                vrun += ((model(xb) - yb) ** 2).mean().item() * xb.size(0)
        val_mse = vrun / len(ds.val)

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "best_val": best_val, **norm.state()}
        torch.save(ck, last_path)
        if val_mse < best_val:
            best_val = val_mse
            ck["best_val"] = best_val
            torch.save(ck, best_path)

        print(f"epoch {epoch:3d}  train {train_mse:.4f}  val {val_mse:.4f}  "
              f"{time.time()-t0:.1f}s")

    gate_d(model, ds, device, best_path)


def gate_d(model, ds, device, best_path):
    """Held-out test error in dB, overall and in the deep-resonance region."""
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = ds.normalizer

    loader = DataLoader(ds.test, batch_size=512, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu())
            trues.append(yb)
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()

    mse_norm = float(((pred - true) ** 2).mean())
    pred_db = norm.to_db(pred)
    true_db = norm.to_db(true)
    ae = np.abs(pred_db - true_db)

    overall_mae = float(ae.mean())
    deep = true_db < -10.0
    deep_mae = float(ae[deep].mean()) if deep.any() else float("nan")
    per_freq = ae.mean(axis=0)                       # (81,)
    worst = np.argsort(per_freq)[-5:][::-1]

    print("\n================ GATE D (test set) ================")
    print(f"normalized MSE          : {mse_norm:.4f}")
    print(f"dB MAE overall          : {overall_mae:.3f} dB")
    print(f"dB MAE deep (<-10 dB)    : {deep_mae:.3f} dB   "
          f"(share of points: {deep.mean()*100:.2f}%)")
    print(f"worst freq bins (idx)   : {worst.tolist()}")
    print(f"worst freq bin MAE      : {per_freq[worst].round(3).tolist()} dB")
    print("Decision: pick a floor BEFORE optimizing. Suggested pilot floor:")
    print("  overall dB MAE < ~1.0 dB AND deep dB MAE < ~2-3 dB.")
    print("  If deep MAE is large, the surrogate is weak exactly where targets")
    print("  live -> expect a big surrogate-vs-EM gap; consider more capacity")
    print("  or reweighting deep samples before trusting GD results.")
    print("===================================================")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/antenna_dataset.mat")
    p.add_argument("--ckpt_dir", default="models")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--deep_weight", type=float, default=0.0,
                   help="extra weight on points with true dB < deep_thr "
                        "(0 = plain MSE; try 10-30 to fix the deep-region blind spot)")
    p.add_argument("--deep_thr", type=float, default=-10.0,
                   help="dB threshold defining 'deep resonance' samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0,
                   help="keep fixed across seeds so the test set never changes")
    p.add_argument("--workers", type=int, default=2,
                   help="set 0 on macOS if you hit multiprocessing errors")
    return p.parse_args()


if __name__ == "__main__":
    run(get_args())
