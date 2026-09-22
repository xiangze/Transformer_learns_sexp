"""train_cont.py -- train a NumTransformer on the continuous S-expression task.

    python train_cont.py --mode eval        --steps 4000 --out ckpt_eval.pt
    python train_cont.py --mode grid        --steps 4000 --out ckpt_grid.pt
    python train_cont.py --mode ignore_code --steps 4000 --out ckpt_ignore.pt
    python train_cont.py --mode eval --steps 0 --out ckpt_rand.pt   # random-init floor

The three data modes are the controls the probe is calibrated against:
  eval        -- genuine higher-order evaluation is possible
  grid        -- v lives on a finite grid, so a (u, v) lookup table suffices
  ignore_code -- the target ignores u entirely
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from model_num import ModelCfg, NumTransformer
from sexp_cont import make_batch, out_positions


def evaluate(model, rng, out_pos, device, mode, n=16, bsz=128):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            b = make_batch(rng, bsz, mode=mode)
            ids = torch.from_numpy(b.ids).to(device)
            vals = torch.from_numpy(b.vals).to(device)
            y = torch.from_numpy(b.y).to(device)
            pred, _ = model(ids, vals, out_pos)
            tot += torch.mean((pred - y) ** 2).item()
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["eval", "grid", "ignore_code"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bsz", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--numemb", default="both", choices=["linear", "fourier", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out_pos = torch.from_numpy(out_positions()).to(device)

    cfg = ModelCfg(
        d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
        d_ff=args.d_ff, numemb=args.numemb,
    )
    model = NumTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))

    t0 = time.time()
    for step in range(args.steps):
        b = make_batch(rng, args.bsz, mode=args.mode)
        ids = torch.from_numpy(b.ids).to(device)
        vals = torch.from_numpy(b.vals).to(device)
        y = torch.from_numpy(b.y).to(device)
        pred, _ = model(ids, vals, out_pos)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % max(args.steps // 10, 1) == 0:
            mse = evaluate(model, rng, out_pos, device, args.mode)
            held = evaluate(model, rng, out_pos, device, "eval")
            print(f"step {step+1:6d}  train {loss.item():.5f}  "
                  f"val[{args.mode}] {mse:.5f}  val[eval] {held:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(),
                "mode": args.mode, "steps": args.steps}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
