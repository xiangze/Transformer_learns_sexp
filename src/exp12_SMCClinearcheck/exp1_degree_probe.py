"""
exp1_degree_probe.py
=====================================================================
EXPERIMENT 1 -- measure the multiplicative multiplicity a Transformer
actually realizes in a designated in-context variable.

Three stages, in increasing order of what can go wrong:

  S0  INSTRUMENT VALIDATION (no network)
      Run the Chebyshev degree estimator on exact polynomials x^p.
      It must return exactly p.  If S0 fails, nothing downstream means
      anything.

  S1  ARCHITECTURE BASELINE (untrained networks)
      Probe randomly initialised models.  LayerNorm and softmax are
      themselves non-linear in x, so an untrained model has non-zero
      apparent degree.  Every trained-model number must be read against
      this floor, otherwise "the model computes x^4" is indistinguishable
      from "LayerNorm is not affine".

  S2  CALIBRATION ON KNOWN GROUND TRUTH (trained networks)
      Train on task m and probe.  For successfully fitted m the input-space
      degree should recover m.  This is what licenses using the probe as a
      measuring device in Experiment 2.
      Also probe the EMBEDDING-space degree, which measures the ceiling:
      the multiplicative capacity available in directions the task does not
      use.

Usage
-----
  python3 exp1_degree_probe.py --preset cpu_smoke
  python3 exp1_degree_probe.py --preset cpu_small
  python3 exp1_degree_probe.py --preset gpu_full --device cuda
"""

import math
import os

import torch

from mult_common import (
    ModelCfg, TrainCfg, MultTransformer, base_argparser, resolve_preset,
    resolve_device, chebyshev_spectrum, effective_degree,
    probe_input_degree, probe_embedding_degree, train_one, save,
    append_jsonl, load_jsonl, output_root,
)


# ---------------------------------------------------------------- S0

def stage0_instrument(nodes=32, max_p=12, tol=1e-6):
    """
    The estimator applied to an exact x^p must return exactly p in the
    noise-free limit.  NOTE: a fixed loose tolerance systematically
    UNDER-estimates high degrees, because the leading Chebyshev coefficient
    of x^p carries a vanishing share of the energy (for x^9 it is ~0.6% of
    the norm).  That is why the probe reports a tolerance CURVE and not a
    single number: the operative tolerance must be matched to the model's
    own fit error, and cannot be chosen a priori.
    """
    j = torch.arange(nodes, dtype=torch.float64)
    t = torch.cos(math.pi * (j + 0.5) / nodes)
    rows, ok_all = [], True
    for p in range(1, max_p + 1):
        c = chebyshev_spectrum(t.pow(p)[None, :], nodes)
        d_exact = effective_degree(c, tol)
        d_loose = effective_degree(c, 1e-2)
        ok = abs(d_exact - p) < 1e-9
        ok_all &= ok
        rows.append(dict(stage="S0", true_degree=p, recovered_tol1e6=d_exact,
                         recovered_tol1e2=d_loose, ok=bool(ok)))
    for name, f in (("exp(3x)", lambda z: torch.exp(3 * z)),
                    ("tanh(8x)", lambda z: torch.tanh(8 * z)),
                    ("1/(1+25x^2)", lambda z: 1 / (1 + 25 * z ** 2))):
        c = chebyshev_spectrum(f(t)[None, :], nodes)
        rows.append(dict(stage="S0", true_degree=name,
                         recovered_tol1e6=effective_degree(c, tol),
                         recovered_tol1e2=effective_degree(c, 1e-2), ok=None))
    return rows, ok_all


# ---------------------------------------------------------------- S1

def stage1_baseline(cfg, args, device):
    recs = []
    for L in cfg["L_list"]:
        for d in (cfg.get("d_map", {}).get(L, None) and [cfg["d_map"][L]]) or cfg["d_list"]:
            mcfg = ModelCfg(n_layer=L, d_model=d, n_head=args.n_head, K=cfg["K"], act=args.act)
            torch.manual_seed(0)
            model = MultTransformer(mcfg).to(device)
            p_in = probe_input_degree(model, mcfg, device)
            p_em = probe_embedding_degree(model, mcfg, device)
            recs.append(dict(stage="S1_untrained", L=L, d=d,
                             n_params=mcfg.n_params(),
                             deg_in={k: v for k, v in p_in.items() if k.startswith("deg@")},
                             deg_emb={k: v for k, v in p_em.items() if k.startswith("deg@")},
                             curve_in=p_in["curve"][:16], curve_emb=p_em["curve"][:16]))
            print(f"  [S1] L={L} d={d:4d}  deg_in@.1={p_in['deg@0.1']:.2f} "
                  f"@.01={p_in['deg@0.01']:.2f}   deg_emb@.1={p_em['deg@0.1']:.2f} "
                  f"@.01={p_em['deg@0.01']:.2f}", flush=True)
    return recs


# ---------------------------------------------------------------- S2

def stage2_calibration(cfg, args, device):
    recs = []
    done = set()
    if args.jsonl:
        prev = load_jsonl(args.jsonl)
        done = {(r.get("m"), r.get("seed")) for r in prev if r.get("stage") == "S2_trained"}
    L = max(cfg["L_list"])
    d = (cfg.get("d_map", {}).get(L) or max(cfg["d_list"]))
    for m in cfg["m_list"]:
        for seed in range(cfg["seeds"]):
            if (m, seed) in done:
                continue
            mcfg = ModelCfg(n_layer=L, d_model=d, n_head=args.n_head,
                            K=cfg["K"], act=args.act)
            tcfg = TrainCfg(steps=cfg["steps"], batch=cfg["batch"],
                            eval_batch=cfg["eval_batch"], seed=seed,
                            amp=cfg.get("amp", False), ood_ratio=args.ood_ratio)
            model, res = train_one(mcfg, tcfg, m, device)
            p_in = probe_input_degree(model, mcfg, device)
            p_em = probe_embedding_degree(model, mcfg, device)
            # tolerance matched to the model's OWN residual: structure below
            # this level is unresolvable, so reading the degree any finer
            # measures fit noise rather than multiplicative multiplicity.
            noise = max(math.sqrt(max(1e-12, 1 - res["r2"])), 1e-3)
            import torch as _t
            curve = _t.tensor(p_in["curve"])
            dn = (curve < noise).float()
            deg_matched = float(dn.argmax().item()) if dn.any() else float(len(curve) - 1)
            rec = dict(stage="S2_trained", m=m, L=L, d=d, seed=seed,
                       n_params=mcfg.n_params(), noise_tol=noise,
                       deg_matched=deg_matched,
                       deg_in={k: v for k, v in p_in.items() if k.startswith("deg@")},
                       deg_emb={k: v for k, v in p_em.items() if k.startswith("deg@")},
                       curve_in=p_in["curve"][:16], curve_emb=p_em["curve"][:16], **res)
            recs.append(rec)
            if args.jsonl:
                append_jsonl(rec, args.jsonl)
            print(f"  [S2] m={m} seed={seed}  R2={res['r2']:.3f} R2ood={res['r2_ood']:.3f}"
                  f"  deg_matched={deg_matched:.0f} (tol={noise:.3f})"
                  f"  deg_emb@.01={p_em['deg@0.01']:.2f}  ({res['wall_s']}s)", flush=True)
    return recs


def main():
    ap = base_argparser("Experiment 1: effective-degree probe")
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()
    cfg = resolve_preset(args)
    device = resolve_device(args.device)
    out = args.out or os.path.join(output_root(), f"exp1_{args.preset}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"device={device}  preset={args.preset}  act={args.act}  out={out}", flush=True)

    s0, s1, s2 = [], [], []
    if "0" in args.stages:
        print("\n== S0: instrument validation ==", flush=True)
        s0, s0_ok = stage0_instrument()
        for r in s0:
            print(f"  true={r['true_degree']:>11}  recovered@1e-6={r['recovered_tol1e6']:>5}"
                  f"  recovered@1e-2={r['recovered_tol1e2']:>5}", flush=True)
        print(f"  instrument OK: {s0_ok}", flush=True)
    else:
        s0_ok = None

    if "1" in args.stages:
        print("\n== S1: untrained architecture baseline ==", flush=True)
        s1 = stage1_baseline(cfg, args, device)

    if "2" in args.stages and not args.skip_train:
        print("\n== S2: trained-model calibration ==", flush=True)
        s2 = stage2_calibration(cfg, args, device)

    save(s0 + s1 + s2, out, meta=dict(preset=args.preset, device=str(device),
                                      act=args.act, eps=args.eps, cfg=str(cfg),
                                      instrument_ok=s0_ok))


if __name__ == "__main__":
    main()
