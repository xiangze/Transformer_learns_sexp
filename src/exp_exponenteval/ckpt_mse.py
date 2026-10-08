"""ckpt_mse.py -- report a checkpoint's held-out MSE, under every data mode.

Use when the training log is gone.  The number that matters for the lookup
control is val[grid]: a small MSE there alongside a large |M-GT| from the probe
is the lookup signature -- correct values at the memorised nodes, wrong local
slopes.  A large MSE instead means the model simply did not converge and
|M-GT| says nothing about lookup structure.

The language must match the checkpoint, so set the same SEXP_* variables used
for training:

    SEXP_N_ANGLE=4 SEXP_MAX_DEPTH=2 python ckpt_mse.py lookup/ckpt_lookup50k.pt
    python ckpt_mse.py ckpt_eval50k.pt
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from probe_mlp import load_model
from sexp_cont import LANG, make_batch, out_positions


def mse(model, out_pos, device, mode, n=40, bsz=256, seed=0):
    rng = np.random.default_rng(seed)
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            b = make_batch(rng, bsz, mode=mode)
            pred, _ = model(torch.from_numpy(b.ids).to(device),
                            torch.from_numpy(b.vals).to(device), out_pos)
            tot += float(torch.mean((pred - torch.from_numpy(b.y).to(device)) ** 2))
    return tot / n


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_pos = torch.from_numpy(out_positions()).to(device)
    print(f"language: {LANG}")
    for path in sys.argv[1:]:
        model, ck = load_model(path, device)
        print(f"\n{path}  (trained mode: {ck.get('mode')}, steps: {ck.get('steps')})")
        for mode in ("eval", "grid", "ignore_code"):
            m = mse(model, out_pos, device, mode)
            # |M-GT| that this MSE alone would explain: for v ~ N(0,I_n),
            # MSE = ||M - Mhat||_F^2 / n, and ||A||_F = sqrt(n) for orthogonal A
            print(f"   val[{mode:11s}] = {m:.6f}   "
                  f"(a pure linear-map error this size means |M-GT| ~ {np.sqrt(m):.4f})")


if __name__ == "__main__":
    main()
