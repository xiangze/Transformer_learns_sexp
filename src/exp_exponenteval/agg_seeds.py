"""agg_seeds.py -- aggregate probe results across training seeds.

Every number in this experiment has so far been n=1. A probe CI covers the
sampling of probe points within one model; it says nothing about whether a
differently seeded training run lands in the same place. That needs several
checkpoints and this script.

    # train the replicates (see make_ckpts.sh SEEDS=...)
    for s in 0 1 2 3 4; do
      python train_cont.py --mode eval --steps 50000 --bsz 512 --lr 1e-3 \
          --seed $s --out ckpt_eval50k_s$s.pt
      python probe_mlp.py --ckpt ckpt_eval50k_s$s.pt --n-u 64 --n-v 16 \
          --n-pairs 64 --json res_eval50k_s$s.json
    done
    python agg_seeds.py res_eval50k_s*.json

Reports, per path condition, the across-seed mean, standard deviation and
range of each statistic.  A verdict that holds in one seed and not another is
not a verdict.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

#: statistics worth tracking across seeds, and how many decimals to print
FIELDS = [("fit", 5), ("relnorm", 3), ("sep_inv", 5), ("sep_excess", 2),
          ("rho_lin", 4), ("rho_excess", 2), ("id_op", 2), ("op_spread", 1)]


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)

    runs = []
    for p in paths:
        with open(p) as f:
            runs.append((p, json.load(f)))

    modes = {r.get("trained_mode") for _, r in runs}
    dists = {r.get("v_dist") for _, r in runs}
    if len(modes) > 1 or len(dists) > 1:
        print(f"REFUSING: these files mix trained modes {modes} or probe "
              f"distributions {dists}. Aggregate like with like.")
        raise SystemExit(1)

    print(f"{len(runs)} seeds, trained mode {modes.pop()}, "
          f"probed at v~{dists.pop()}")
    for p, _ in runs:
        print(f"   {p}")

    by_path = defaultdict(lambda: defaultdict(list))
    for _, r in runs:
        for path, row in r["paths"].items():
            for k, _ in FIELDS:
                if k in row and row[k] is not None:
                    by_path[path][k].append(row[k])

    for path, stats in by_path.items():
        print(f"\n--- path={path} ---")
        for k, nd in FIELDS:
            v = np.asarray([x for x in stats.get(k, []) if np.isfinite(x)],
                           dtype=float)
            if v.size == 0:
                print(f"   {k:11s} : (no finite values)")
                continue
            flag = ""
            if k.endswith("excess") and v.size > 1:
                # a verdict that straddles 1 or 3 across seeds is not a verdict
                if v.min() < 1 < v.max():
                    flag = "   <- straddles 1 across seeds"
                if v.min() < 3 < v.max():
                    flag = "   <- straddles 3 across seeds"
            print(f"   {k:11s} : mean {v.mean():.{nd}f}  sd {v.std(ddof=1) if v.size > 1 else 0:.{nd}f}"
                  f"  range [{v.min():.{nd}f}, {v.max():.{nd}f}]  n={v.size}{flag}")

    ffn = [r.get("ffn_eta2_u") for _, r in runs if r.get("ffn_eta2_u") is not None]
    if ffn:
        f = np.asarray(ffn, dtype=float)
        print(f"\n[localization only] ffn_eta2_u : mean {f.mean():.4f}  "
              f"range [{f.min():.4f}, {f.max():.4f}]")


if __name__ == "__main__":
    main()
