"""
exp2_multiplicity_sweep.py
=====================================================================
EXPERIMENT 2 -- the actual falsification test.

Sweep (m, L, d):  m = multiplicative multiplicity of one in-context
variable, L = depth, d = width.  Everything else (sequence length, token
count, additive reduction over the a_i, target variance) is held fixed
across m by construction, so m is the only thing that changes.

    m*(L, d) = largest m, contiguous from 1, with test R^2 >= threshold

Decision rule
-------------
    m* grows with L, flat in d      -> depth-graded !_L  (thesis claim C survives)
    m* grows with d, flat in L      -> width-graded !_d  (claim C FALSIFIED;
                                        contraction is paid in residual-stream
                                        dimensions, as in Tracr's construction)
    m* grows with neither, no ceiling -> unrestricted contraction (claim B FALSIFIED)
    m* tracks n_params only          -> no resource structure beyond capacity;
                                        the categorical reading adds nothing here

The iso-parameter presets hold L * d^2 ~ const so that "depth helps" cannot
be confused with "more parameters help".  Without that control the sweep is
uninterpretable, because increasing L at fixed d also increases capacity.

Note on m = 1
-------------
m = 1 is NOT a trivial baseline: it is the condition in which x is consumed
K times ADDITIVELY.  Vect has a diagonal for the biproduct, and frozen-softmax
attention Z = AV implements exactly that fan-out.  So m = 1 should succeed
everywhere.  The interesting quantity is the gap between m = 1 and m >= 2,
which is the gap between (+)-copy (free) and (x)-copy (the thing linear logic
forbids without !).

Usage
-----
  python3 exp2_multiplicity_sweep.py --preset cpu_smoke
  python3 exp2_multiplicity_sweep.py --preset cpu_small
  python3 exp2_multiplicity_sweep.py --preset gpu_full --device cuda
  python3 exp2_multiplicity_sweep.py --preset gpu_iso  --device cuda
  python3 exp2_multiplicity_sweep.py --analyze results.json      # re-analyse only
"""

import json
import math
import os
import sys
from collections import defaultdict

import torch

from mult_common import (
    ModelCfg, TrainCfg, base_argparser, resolve_preset, resolve_device,
    train_one, probe_input_degree, probe_embedding_degree, save,
    append_jsonl, load_jsonl, parse_shard, output_root,
)


def configs(cfg, args):
    for L in cfg["L_list"]:
        ds = [cfg["d_map"][L]] if "d_map" in cfg else cfg["d_list"]
        for d in ds:
            for m in cfg["m_list"]:
                for seed in range(cfg["seeds"]):
                    yield L, d, m, seed


def run(cfg, args, device, out):
    done = set()
    recs = []
    if args.jsonl:
        recs = load_jsonl(args.jsonl)
        done = {(r["L"], r["d"], r["m"], r["seed"]) for r in recs}
        if done:
            print(f"[resume] {len(done)} configs already done", flush=True)
    k, nsh = parse_shard(args.shard)
    total = sum(1 for _ in configs(cfg, args))
    for i, (L, d, m, seed) in enumerate(configs(cfg, args), 1):
        if (i - 1) % nsh != k:
            continue
        if (L, d, m, seed) in done:
            continue
        mcfg = ModelCfg(n_layer=L, d_model=d, n_head=args.n_head,
                        K=cfg["K"], act=args.act)
        tcfg = TrainCfg(steps=cfg["steps"], batch=cfg["batch"],
                        eval_batch=cfg["eval_batch"], seed=seed,
                        amp=cfg.get("amp", False), ood_ratio=args.ood_ratio)
        model, res = train_one(mcfg, tcfg, m, device)
        p_in = probe_input_degree(model, mcfg, device)
        p_em = probe_embedding_degree(model, mcfg, device)
        # tolerance matched to the model's own residual (see exp1 S2)
        noise = max(math.sqrt(max(1e-12, 1 - res["r2"])), 1e-3)
        curve = p_in["curve"]
        deg_matched = next((i for i, v in enumerate(curve) if v < noise), len(curve) - 1)
        rec = dict(L=L, d=d, m=m, seed=seed, n_params=mcfg.n_params(),
                   deg_matched=float(deg_matched), noise_tol=noise,
                   degmatch=float(abs(deg_matched - m) <= 0.5),
                   deg_in={k: v for k, v in p_in.items() if k.startswith("deg@")},
                   deg_emb={k: v for k, v in p_em.items() if k.startswith("deg@")},
                   curve_in=p_in["curve"][:16], **res)
        recs.append(rec)
        print(f"[{i:4d}/{total}] L={L} d={d:4d} m={m:2d} s={seed}  "
              f"R2={res['r2']:+.3f}  R2ood={res['r2_ood']:+.3f}  "
              f"({res['wall_s']}s)", flush=True)
        if args.jsonl:
            append_jsonl(rec, args.jsonl)
        if not args.light_save:
            save(recs, out, meta=dict(preset=args.preset, device=str(device),
                                  act=args.act, threshold=args.threshold,
                                  cfg=str(cfg)))
    return recs


# ---------------------------------------------------------------- analysis

def m_star(rows, threshold, metric="r2_ood_matched"):
    """Largest m, contiguous from m=1, with median metric >= threshold."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["m"]].append(r[metric])
    best = 0
    for m in sorted(by_m):
        med = sorted(by_m[m])[len(by_m[m]) // 2]
        if med >= threshold:
            best = m
        else:
            break
    return best, {m: round(sorted(v)[len(v) // 2], 4) for m, v in sorted(by_m.items())}


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    b = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy ** 2 / (sxx * syy)) if syy > 0 else 1.0
    return dict(slope=round(b, 4), intercept=round(my - b * mx, 4), r2=round(r2, 4))


def analyze(recs, threshold, metric="r2_ood_matched"):
    cells = defaultdict(list)
    for r in recs:
        cells[(r["L"], r["d"])].append(r)

    table = {}
    for (L, d), rows in sorted(cells.items()):
        ms, per_m = m_star(rows, threshold, metric)
        _, r2_in = m_star(rows, threshold, "r2")
        _, r2_fixed = m_star(rows, threshold, "r2_ood") if "r2_ood" in rows[0] else (0, {})
        _, degs = m_star(rows, threshold, "deg_matched")
        table[f"L{L}_d{d}"] = dict(L=L, d=d, m_star=ms,
                                   n_params=rows[0]["n_params"],
                                   metric_by_m=per_m, r2_in_by_m=r2_in,
                                   r2_ood_fixedband_by_m=r2_fixed,
                                   deg_matched_by_m=degs)

    pts = list(table.values())
    out = dict(threshold=threshold, metric=metric, cells=table)

    # marginal scalings, and the parameter-count confound
    out["fit_m*_vs_logL"] = linfit([math.log2(p["L"]) for p in pts],
                                   [p["m_star"] for p in pts])
    out["fit_m*_vs_logd"] = linfit([math.log2(p["d"]) for p in pts],
                                   [p["m_star"] for p in pts])
    out["fit_m*_vs_logparams"] = linfit([math.log2(p["n_params"]) for p in pts],
                                        [p["m_star"] for p in pts])

    # depth at fixed width, and width at fixed depth -- the actual discriminator
    by_d, by_L = defaultdict(list), defaultdict(list)
    for p in pts:
        by_d[p["d"]].append(p)
        by_L[p["L"]].append(p)
    out["depth_effect_at_fixed_width"] = {
        f"d{d}": linfit([math.log2(q["L"]) for q in v], [q["m_star"] for q in v])
        for d, v in sorted(by_d.items()) if len(v) > 1}
    out["width_effect_at_fixed_depth"] = {
        f"L{L}": linfit([math.log2(q["d"]) for q in v], [q["m_star"] for q in v])
        for L, v in sorted(by_L.items()) if len(v) > 1}

    # iso-parameter families: same n_params, different (L, d)
    iso = defaultdict(list)
    for p in pts:
        iso[round(math.log2(max(1, p["n_params"])) * 2) / 2].append(p)
    out["iso_param_families"] = {
        str(k): sorted([(q["L"], q["d"], q["m_star"]) for q in v])
        for k, v in sorted(iso.items()) if len(v) > 1}

    fL = out["depth_effect_at_fixed_width"]
    fd = out["width_effect_at_fixed_depth"]
    sL = sum(v["slope"] for v in fL.values()) / max(1, len(fL))
    sd = sum(v["slope"] for v in fd.values()) / max(1, len(fd))
    out["mean_slope_per_doubling"] = dict(depth=round(sL, 3), width=round(sd, 3))
    if max(abs(sL), abs(sd)) < 0.25:
        verdict = ("no_resource_scaling: m* is flat in both depth and width over "
                   "the swept range -- either the ceiling lies outside the grid, "
                   "or optimisation rather than expressivity is binding")
    elif sL > 2 * abs(sd):
        verdict = "depth_graded: supports !_L (thesis claim C survives)"
    elif sd > 2 * abs(sL):
        verdict = "width_graded: FALSIFIES !_L; contraction is paid in width (Tracr-style)"
    else:
        verdict = "mixed: depth and width contribute comparably; !_L under-determined"
    out["verdict"] = verdict
    return out


def main():
    ap = base_argparser("Experiment 2: multiplicity x depth x width sweep")
    ap.add_argument("--analyze", default=None, help="analyse an existing json and exit")
    ap.add_argument("--metric", default="r2_ood_matched",
                    choices=["r2_ood_matched", "r2_ood", "r2", "degmatch"],
                    help="success criterion. r2_ood (default) requires the model to "
                         "EXTRAPOLATE, which is the only way to distinguish realizing "
                         "x^m from approximating it on the training support -- exp1 S2 "
                         "shows in-distribution R^2 stays ~1.000 up to m=10 and would "
                         "give a spurious 'contraction is free' verdict.")
    args = ap.parse_args()

    if args.analyze:
        if args.analyze.endswith(".jsonl"):
            recs = load_jsonl(args.analyze)
        else:
            recs = json.load(open(args.analyze))["records"]
        res = analyze(recs, args.threshold, args.metric)
        print(json.dumps(res, indent=1))
        p = args.analyze.rsplit(".", 1)[0] + "_analysis.json"
        json.dump(res, open(p, "w"), indent=1)
        print(f"[saved] {p}")
        return

    cfg = resolve_preset(args)
    device = resolve_device(args.device)
    tag = args.shard.replace("/", "of")
    out = args.out or os.path.join(output_root(), f"exp2_{args.preset}_{tag}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"device={device}  preset={args.preset}  act={args.act}  "
          f"shard={args.shard}  out={out}", flush=True)
    recs = run(cfg, args, device, out)
    res = analyze(recs, args.threshold, args.metric)
    print("\n== analysis ==")
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
    json.dump(res, open(out.replace(".json", "_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
