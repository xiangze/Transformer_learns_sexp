#!/usr/bin/env python3
"""
exp3_eval_probe.py -- runner.

  --stages g   gauge suite G1-G4 (G4 is where the expansion point is judged)
  --stages e   expansion-point comparison: does centroid shrinkage + jitter
               averaging actually stabilise the operating point?
  --stages s   the (l_in, l_out) scan
  --stages p   position transfer and FV portability

With no --hf-model it runs on the numpy reference model, so everything works
with no GPU and no checkpoint. With --hf-model it uses eval_torch.TorchProbe,
which runs the same code on cpu and cuda; the analysis functions take the probe
protocol, not the model.

Typical CUDA run of the scan, in three steps -- build the frames once, shard the
cells over the GPUs, reassemble:

  python3 exp3_eval_probe.py --stages s --hf-model <ckpt> --device cuda \
      --dtype bf16 --k 16 --n-u 64 --n-v 8 --batch-size 512 \
      --cache scan.jsonl --shard 0/4
"""

import argparse
import sys

import numpy as np

from eval_common import jsonl_append
from eval_probe import (TinyLM, TinyProbe, ExpansionPoint, build_in_basis,
                        readout_frame, check_G1, check_G2, check_G3, check_G4,
                        _probe_for)
from eval_layers import (scan2d, format_scan, apply_band, position_transfer,
                         fv_transfer_matrix)


# ------------------------------------------------------------------ probes

def build_numpy_probe(model, task, layers_in):
    flat = [r[0] for r in task["prompts"]]
    B = {l: build_in_basis(model, flat, l, task["x_pos"], task["value_ids"])
         for l in layers_in}
    return TinyProbe(model, B, readout_frame(model, task["value_ids"]))


def build_torch_probe(args, task, layers_in):
    """TorchProbe over an HF checkpoint. Weights may be bf16; the Jacobian is
    always taken in fp32 -- central differences at eps ~ 1e-3 on bf16
    activations measure rounding, not M."""
    import torch
    from transformers import AutoModelForCausalLM
    from eval_torch import hf_spec, TorchProbe, build_in_basis as bi_torch

    wdtype = dict(bf16=torch.bfloat16, fp16=torch.float16,
                  fp32=torch.float32)[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=wdtype,
        device_map=args.device if args.device != "auto" else "auto")
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    spec = hf_spec(model)
    dev = next(model.parameters()).device

    flat = [r[0] for r in task["prompts"]]
    B = {l: bi_torch(spec, flat, l, task["x_pos"], task["value_ids"],
                     device=dev, dtype=torch.float32)
         for l in layers_in}
    return TorchProbe(spec, task["value_ids"], B, device=str(dev),
                      dtype=torch.float32, eps=args.fd_eps,
                      batch_size=args.batch_size)


def make_probe(args, model, task, layers_in):
    if args.hf_model:
        return build_torch_probe(args, task, layers_in)
    return build_numpy_probe(model, task, layers_in)


# ------------------------------------------------------------------ task

def make_task(model, n_u=24, n_v=6, k=8, T=12, x_pos=9, read_pos=11):
    """Fixed-layout artificial prompts. Replace with the S-expression grammar;
    only codes_u / prompts / x_pos / read_pos / value_ids are used downstream."""
    value_ids = list(range(8, 8 + k))

    def ids_of(u, v, xp=x_pos):
        s = np.full(T, 2, dtype=int)
        s[0], s[2], s[3] = 1, 20 + u, 40
        s[xp] = value_ids[v]
        s[-1] = 3
        return s

    prompts = [[ids_of(u, v) for v in range(n_v)] for u in range(n_u)]
    blank = [[np.where(np.arange(T) == 2, 2, ids_of(u, v)) for v in range(n_v)]
             for u in range(n_u)]
    return dict(prompts=prompts, blank=blank, ids_of=ids_of,
                value_ids=value_ids, x_pos=x_pos, read_pos=read_pos,
                codes_u=list(range(n_u)), k=k)


def make_task_hf(args, tok, n_v=None):
    """Fixed-layout prompts over a real tokenizer.

    Every template must tokenise to the SAME length with the x-slot and the
    readout at the SAME index: A(u) and the injection frame are only comparable
    across u under a fixed layout. Single-token value and function names make
    that automatic; with a real grammar, pad to a common length and assert the
    positions rather than hoping.
    """
    k, n_u = args.k, args.n_u
    n_v = n_v or args.n_v
    vals = [f" v{i}" for i in range(k)]
    fns = [f" f{i}" for i in range(n_u)]
    value_ids = []
    for s in vals:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"value name {s!r} is not a single token ({t}); "
                             "pick names your tokenizer keeps atomic")
        value_ids.append(t[0])

    head = tok.encode("( apply", add_special_tokens=False)
    tail = tok.encode(" ) =>", add_special_tokens=False)

    def ids_of(u, v, xp=None):
        f = tok.encode(fns[u], add_special_tokens=False)
        ids = head + f + [value_ids[v]] + tail
        return np.asarray(ids)

    probe_ids = ids_of(0, 0)
    x_pos = len(head) + len(tok.encode(fns[0], add_special_tokens=False))
    read_pos = len(probe_ids) - 1
    lens = {len(ids_of(u, v)) for u in range(n_u) for v in range(n_v)}
    if len(lens) != 1:
        raise ValueError(f"templates tokenise to different lengths {lens}: "
                         "fix the layout before scanning")

    prompts = [[ids_of(u, v) for v in range(n_v)] for u in range(n_u)]
    blank_tok = tok.encode(" _", add_special_tokens=False)[0]

    def blank_of(u, v):
        ids = ids_of(u, v).copy()
        ids[len(head)] = blank_tok
        return ids

    blank = [[blank_of(u, v) for v in range(n_v)] for u in range(n_u)]
    return dict(prompts=prompts, blank=blank, ids_of=ids_of,
                value_ids=value_ids, x_pos=x_pos, read_pos=read_pos,
                codes_u=list(range(n_u)), k=k)


# ------------------------------------------------------------------ stages

def stage_gauge(args, model, task):
    print("\n=== gauge suite ===\n")
    rng = np.random.default_rng(args.seed)
    l_in, l_out = args.l_in, args.l_out
    pr = make_probe(args, model, task, [l_in])
    ep = ExpansionPoint(task["k"], mode=args.mode, alpha=args.alpha,
                        n_jitter=1, seed=args.seed)
    ids = task["prompts"][0][0]
    flat = [r[0] for r in task["prompts"]]
    rows = [check_G1(pr, ids, task["x_pos"], task["read_pos"], l_in, l_out, ep, rng),
            check_G2(model, flat, ids, task["x_pos"], task["read_pos"], l_in,
                     l_out, task["value_ids"], ep, rng),
            check_G3(model, flat, ids, task["x_pos"], task["read_pos"], l_in,
                     l_out, task["value_ids"], ep, rng)]
    ep4 = ExpansionPoint(task["k"], mode=args.mode, alpha=args.alpha,
                         n_jitter=args.n_jitter, sigma=args.sigma, seed=args.seed)
    rows.append(check_G4(pr, task["prompts"], task["x_pos"], task["read_pos"],
                         l_in, l_out, ep4, n_rep=args.n_rep,
                         rep_sigma=args.rep_sigma * args.alpha, seed=args.seed))
    for r in rows:
        print(" ", {k: (round(v, 10) if isinstance(v, float) else v)
                    for k, v in r.items()})
        jsonl_append(args.jsonl, dict(key=f"gauge/{r['check']}", **r))
    return 0


def stage_expansion(args, model, task):
    """Does option (b) fix the operating-point instability that G4 flagged?"""
    print("\n=== expansion-point comparison (G4 statistic) ===\n")
    pr = make_probe(args, model, task, [args.l_in])
    print(f"  {'mode':>10} {'alpha':>6} {'jit':>4} {'sigma':>6} "
          f"{'pr_mean':>8} {'pr_cv':>7} {'id_mean':>8} {'id_cv':>7}  ok")
    grid = [("onehot", 1.0, 1, 0.0),
            ("onehot", 1.0, args.n_jitter, args.sigma),
            ("shrink", 0.75, args.n_jitter, args.sigma),
            ("shrink", 0.50, args.n_jitter, args.sigma),
            ("shrink", 0.25, args.n_jitter, args.sigma),
            ("shrink", 0.10, args.n_jitter, args.sigma)]
    for mode, alpha, jit, sig in grid:
        ep = ExpansionPoint(task["k"], mode=mode, alpha=alpha, n_jitter=jit,
                            sigma=sig, seed=args.seed)
        # the v-dependent part of c0 has size alpha, so a fixed absolute
        # displacement is relatively larger for small alpha. Scale it with
        # alpha or the comparison is rigged against shrinkage.
        r = check_G4(pr, task["prompts"], task["x_pos"], task["read_pos"],
                     args.l_in, args.l_out, ep, n_rep=args.n_rep,
                     rep_sigma=args.rep_sigma * alpha, seed=args.seed)
        print(f"  {mode:>10} {alpha:6.2f} {jit:4d} {sig:6.3f} "
              f"{r['pr_mean']:8.2f} {r['pr_cv']:7.3f} {r['id_mean']:8.2f} "
              f"{r['id_cv']:7.3f}  {'yes' if r['passed'] else 'NO'}")
        jsonl_append(args.jsonl, dict(key=f"expansion/{mode}/{alpha}/{jit}", **r))
    print("\n  pr_cv < 0.15 is the precondition for reading r_op at all.")
    return 0


def _layer_grid(n_layers, args):
    """--layer-step thins the grid: a 32-layer model has 528 cells, and at
    minutes per cell that is not a first run. Step 4 gives 36."""
    st = max(1, args.layer_step)
    lin = list(range(0, n_layers, st))
    lout = sorted(set(list(range(0, n_layers + 1, st)) + [n_layers]))
    return lin, lout


def stage_scan(args, model, task):
    print("\n=== (l_in, l_out) scan ===\n")
    n_layers = None
    if args.hf_model:
        from transformers import AutoConfig
        n_layers = AutoConfig.from_pretrained(args.hf_model).num_hidden_layers
    else:
        n_layers = model.L
    layers_in, layers_out = _layer_grid(n_layers, args)
    pr = make_probe(args, model, task, layers_in)
    ep = ExpansionPoint(task["k"], mode=args.mode, alpha=args.alpha,
                        n_jitter=args.n_jitter, sigma=args.sigma, seed=args.seed)

    cells = [(a, c) for a in layers_in for c in layers_out if c > a]
    if args.shard:
        r, n = (int(x) for x in args.shard.split("/"))
        cells = [c for i, c in enumerate(cells) if i % n == r]
        print(f"  shard {r}/{n}: {len(cells)} cells")

    t0 = __import__("time").time()
    res = scan2d(pr, task["prompts"], task["x_pos"], task["read_pos"], ep,
                 layers_in=layers_in, layers_out=layers_out,
                 p_true=args.p, cache=args.cache, cells=cells,
                 progress=lambda a, c: print(
                     f"  cell ({a},{c}) done  {__import__('time').time()-t0:.0f}s",
                     flush=True))
    for key, fmt in (("sep_full", "{:8.2f}"), ("id_twonn", "{:8.2f}"),
                     ("spread", "{:8.3f}")):
        print(format_scan(res, key, fmt), "\n")
    band = apply_band(res)
    print("  apply band (Sep < 0.25, off-diagonal):",
          band if band else "empty -- no apply depth")
    jsonl_append(args.jsonl, dict(key="scan2d",
                                  sep=res["sep_full"].tolist(),
                                  id=res["id_twonn"].tolist(),
                                  layers_in=res["layers_in"],
                                  layers_out=res["layers_out"]))
    return 0


def stage_pos_fv(args, model, task):
    print("\n=== position transfer / FV portability ===\n")
    pr = make_probe(args, model, task, [args.l_in])
    ep = ExpansionPoint(task["k"], mode=args.mode, alpha=args.alpha,
                        n_jitter=args.n_jitter, sigma=args.sigma, seed=args.seed)
    r = position_transfer(pr, lambda u, p: task["ids_of"](u, 0, p),
                          [7, 8, 9], task["read_pos"], args.l_in, args.l_out,
                          ep, task["codes_u"][:8])
    print("  position:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in r.items()})
    if args.hf_model:
        print("  FV: skipped -- head_out is not wired for HF models "
              "(needs the o_proj input split per head); position transfer "
              "above is valid.")
        return 0
    f = fv_transfer_matrix(pr, task["prompts"][:8], task["blank"][:8],
                           task["x_pos"], task["read_pos"], ep,
                           args.l_in, args.l_out)
    print("  FV gain matrix (src x dst):")
    print(np.round(f["gain"], 3))
    print("  best:", f["best"])
    return 0


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="ges")
    ap.add_argument("--mode", default="shrink", choices=["shrink", "onehot", "centroid"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-jitter", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-rep", type=int, default=6)
    ap.add_argument("--rep-sigma", type=float, default=0.3,
                    help="G4 operating-point displacement, in units of alpha")
    ap.add_argument("--l-in", type=int, default=0)
    ap.add_argument("--l-out", type=int, default=4)
    ap.add_argument("--p", type=int, default=None, help="true dim of the f family")
    ap.add_argument("--n-u", type=int, default=24)
    ap.add_argument("--n-v", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--cache", default=None,
                    help="per-cell jsonl; completed cells are skipped on rerun")
    ap.add_argument("--shard", default=None, metavar="RANK/N",
                    help="compute only cells with index %% N == RANK")
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="weight dtype; the Jacobian is always fp32")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    args = ap.parse_args()

    if args.hf_model and "g" in args.stages:
        raise SystemExit(
            "stage g is numpy-only: G2 and G3 rebuild the reference model with "
            "a rotated residual stream / permuted heads, which needs write "
            "access to the weights in a known layout. Run stage g on the "
            "reference model to validate the instrument, then run stages e and "
            "s on the checkpoint.")
    model = TinyLM(V=64, d=32, L=args.layers, H=4, T=24, seed=args.seed)
    task = make_task(model, n_u=args.n_u, n_v=args.n_v, k=args.k)
    if args.hf_model:
        from transformers import AutoTokenizer
        task = make_task_hf(args, AutoTokenizer.from_pretrained(args.hf_model))
    else:
        args.l_out = min(args.l_out, model.L)

    fn = dict(g=stage_gauge, e=stage_expansion, s=stage_scan, p=stage_pos_fv)
    rc = 0
    for s in args.stages.replace(",", ""):
        rc |= fn[s](args, model, task)
    return rc


if __name__ == "__main__":
    sys.exit(main())
