"""
analyze_gpu_exp1.py
=====================================================================
Analysis of GPU stage A (exp1 S2 recalibration, L=4 d=256, 30k steps).

The central question stage A was meant to settle: when the model's own fit
error drops by an order of magnitude, how far does the degree probe resolve?

The decisive move is NOT to read deg_matched off each run.  It is to compare
the OBSERVED truncation curve against the curve an EXACT x^m would produce.
The probe's apparent ceiling and the model's actual ceiling are different
things, and only the reference curve separates them:

  - if observed curve == exact curve down to the model's noise floor,
    the model realizes x^m faithfully and the "ceiling" is the estimator's
  - if observed departs from exact ABOVE the noise floor, that departure
    degree is a real limit on realized multiplicity
"""
import json
import math
import sys

import numpy as np

NODES = 32
PATH = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/attachment.txt"


def cheb_coeffs(vals, nodes=NODES):
    """Discrete Chebyshev transform at Chebyshev-Gauss nodes (same as the probe)."""
    N = nodes
    j = np.arange(N, dtype=float)
    k = np.arange(N, dtype=float)
    M = np.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / N)
    c = (2.0 / N) * (vals @ M.T)
    c[..., 0] /= 2
    return c


def trunc_curve(c):
    """tail[p] = ||c_{>p}|| / ||c_{>=1}||  -- identical definition to the probe."""
    e = c[1:] ** 2
    tot = max(e.sum(), 1e-300)
    tail = np.cumsum(e[::-1])[::-1] / tot
    return np.sqrt(np.clip(tail, 0, None))


def exact_curve(m, nodes=NODES):
    j = np.arange(nodes, dtype=float)
    t = np.cos(math.pi * (j + 0.5) / nodes)
    return trunc_curve(cheb_coeffs(t ** m))


def deg_at(curve, tol):
    idx = np.nonzero(curve < tol)[0]
    return int(idx[0]) if len(idx) else len(curve)


recs = [json.loads(l) for l in open(PATH) if l.strip()]
by_m = {}
for r in recs:
    by_m.setdefault(r["m"], []).append(r)

print("=" * 96)
print("A. ESTIMATOR BEHAVIOUR: what each read-out gives, and what an EXACT x^m would give")
print("=" * 96)
print(f"{'m':>3} {'deg_matched':>22} {'deg@0.01':>14} {'exact@0.01':>11} "
      f"{'deg@0.03':>12} {'exact@0.03':>11}")
for m in sorted(by_m):
    rs = by_m[m]
    ex = exact_curve(m)
    dm = "/".join(f"{int(r['deg_matched'])}" for r in rs)
    d01 = "/".join(f"{int(r['deg_in']['deg@0.01'])}" for r in rs)
    d03 = "/".join(f"{int(r['deg_in']['deg@0.03'])}" for r in rs)
    print(f"{m:>3} {dm:>22} {d01:>14} {deg_at(ex,0.01):>11} {d03:>12} {deg_at(ex,0.03):>11}")

print()
print("=" * 96)
print("B. FIDELITY: observed truncation curve vs the exact x^m reference")
print("   noise_floor = plateau of the observed curve (last 4 entries)")
print("   departure   = lowest degree p where |obs - exact| exceeds the noise floor")
print("                 ('>=15' = curves agree everywhere the probe can see)")
print("=" * 96)
print(f"{'m':>3} {'seed':>5} {'noise_floor':>12} {'departure':>10} "
      f"{'max|obs-exact| above floor':>28} {'R2':>9} {'R2ood_m':>9}")
summary = []
for m in sorted(by_m):
    ex = exact_curve(m)[:16]
    for r in sorted(by_m[m], key=lambda z: z["seed"]):
        obs = np.array(r["curve_in"][:16])
        floor = float(np.median(obs[-4:]))
        diff = np.abs(obs - ex)
        # only degrees where the exact curve is still above the model's noise
        # floor carry information; below it the probe cannot distinguish
        visible = ex > floor
        dep = "n/a"
        maxd = 0.0
        if visible.any():
            bad = np.nonzero(visible & (diff > floor))[0]
            maxd = float(diff[visible].max())
            dep = str(int(bad[0])) if len(bad) else f">={int(np.nonzero(visible)[0][-1])+1}"
        print(f"{m:>3} {r['seed']:>5} {floor:>12.2e} {dep:>10} {maxd:>28.2e} "
              f"{r['r2']:>9.5f} {r['r2_ood_matched']:>9.3f}")
        summary.append((m, r["seed"], floor, dep, maxd))

print()
print("=" * 96)
print("C. PER-DEGREE COMPARISON for the highest m (is x^16 actually realized?)")
print("=" * 96)
for m in [8, 12, 16]:
    if m not in by_m:
        continue
    ex = exact_curve(m)[:16]
    obs = np.array(by_m[m][0]["curve_in"][:16])
    floor = float(np.median(obs[-4:]))
    print(f"\n  m={m}  (seed 0, model noise floor ~{floor:.1e})")
    print("   deg :" + "".join(f"{p:>9}" for p in range(0, 14)))
    print("   exact:" + "".join(f"{v:>9.2e}" for v in ex[:14]))
    print("   obs  :" + "".join(f"{v:>9.2e}" for v in obs[:14]))

print()
print("=" * 96)
print("D. SEED VARIANCE")
print("=" * 96)
for m in sorted(by_m):
    rs = by_m[m]
    if len(rs) < 2:
        continue
    o = [r["r2_ood_matched"] for r in rs]
    f = [r["r2_ood"] for r in rs]
    flag = "  <-- unstable" if (max(o) - min(o)) > 0.15 or min(f) < 0 else ""
    print(f"  m={m:>2}  R2ood_matched {o[0]:+.3f} / {o[1]:+.3f}   "
          f"R2ood_fixed {f[0]:+.3f} / {f[1]:+.3f}{flag}")
