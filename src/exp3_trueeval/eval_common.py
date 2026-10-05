"""
eval_common.py -- gauge-invariant statistics for "does Phi factor as eval o ([[.]] (x) id)?"

Metrics only (numpy). Extraction lives in eval_probe.py / eval_extract.py.

    Phi(u, v) = y          u = code of f, v = code of x
    M(u, v)   = dy/dv      estimator of [[u]]
    Sep       does Phi factor at all?          M must not depend on v
    r_op      how rich is the image of [[.]]?  effective/intrinsic dim of {M(u)}
    rho       is the image a representation?   M(f o g) =? M(f) M(g)
"""

import json
import math
import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ Sep

def variance_decomposition(T):
    """Two-way decomposition of a grid T of shape (N_u, N_v, ...).

    sep_full = (V_v + V_int) / V_u is the operative statistic. A table indexed
    by the PAIR (u, v) has no v main effect -- it has a large INTERACTION, so
    Var_v/Var_u alone would let it pass.
    """
    T = np.asarray(T, dtype=np.float64)
    Nu, Nv = T.shape[0], T.shape[1]
    X = T.reshape(Nu, Nv, -1)
    grand = X.mean(axis=(0, 1), keepdims=True)
    mu_u = X.mean(axis=1, keepdims=True)
    mu_v = X.mean(axis=0, keepdims=True)
    V_u = float(((mu_u - grand) ** 2).sum() / Nu)
    V_v = float(((mu_v - grand) ** 2).sum() / Nv)
    V_int = float(((X - mu_u - mu_v + grand) ** 2).sum() / (Nu * Nv))
    V_tot = float(((X - grand) ** 2).sum() / (Nu * Nv))
    return dict(V_u=V_u, V_v=V_v, V_int=V_int, V_tot=V_tot,
                sep_raw=V_v / (V_u + EPS),
                sep_full=(V_v + V_int) / (V_u + EPS),
                frac_u=V_u / (V_tot + EPS),
                frac_v=V_v / (V_tot + EPS),
                frac_int=V_int / (V_tot + EPS))


# ------------------------------------------------------------------ E1

def effective_rank(M_u, center=True, energy=0.99):
    """LINEAR effective rank. Scale-dependent: for a p-parameter family traced
    over a wide range it overshoots p by several times (measured: p=5 gives
    pr=25 at theta-width 0.3). Read intrinsic_dim_twonn instead; keep this for
    the noise floor and the spectrum."""
    X = np.asarray(M_u, dtype=np.float64).reshape(len(M_u), -1)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    tot = s2.sum()
    if tot < EPS:
        return dict(pr=0.0, r_energy=0, svals=[], total_energy=0.0)
    pr = float(tot ** 2 / (s2 ** 2).sum())
    r_energy = int(np.searchsorted(np.cumsum(s2) / tot, energy) + 1)
    return dict(pr=pr, r_energy=r_energy, svals=(s / (s[0] + EPS)).tolist(),
                total_energy=float(tot))


def intrinsic_dim_twonn(M_u, discard=0.1):
    """TwoNN (Facco et al. 2017). Recovers the true p across parameter ranges
    where the SVD rank does not."""
    X = np.asarray(M_u, dtype=np.float64).reshape(len(M_u), -1)
    N = len(X)
    if N < 10:
        return dict(id_twonn=float("nan"), n=N)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    r1, r2 = Ds[:, 0], Ds[:, 1]
    ok = r1 > EPS
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    keep = int(n * (1 - discard))
    mu, F = mu[:keep], np.arange(1, keep + 1) / n
    x = np.log(mu + EPS)
    y = -np.log(np.clip(1 - F, EPS, None))
    return dict(id_twonn=float((x @ y) / (x @ x + EPS)), n=N)


# ------------------------------------------------------------------ E2

def functoriality_residual(Mmap, bmap, triples, n_null=200, rng=None,
                           markov=False, beta=1.0):
    """rho with a null from mismatched composites.

    Affine composition, LRE parameterisation:
        (M_f, b_f) o (M_g, b_g) = (M_f M_g, M_f b_g + b_f)

    beta: if the M were scaled by the LRE steepness constant beta (> 1), the
    product picks up beta^2 while the target carries only beta. Pass the same
    beta used for the estimates and the predicted product is divided by it.
    Leave at 1.0 for unscaled Jacobians.
    """
    rng = rng or np.random.default_rng(0)
    affine = bmap is not None

    def compose(kf, kg):
        P = (Mmap[kf] @ Mmap[kg]) / beta
        if markov:
            P = np.clip(P, 0, None)
            P = P / (P.sum(axis=-1, keepdims=True) + EPS)
        if not affine:
            return P, None
        return P, (Mmap[kf] @ bmap[kg]) / beta + bmap[kf]

    def resid(kf, kg, kfg):
        P, q = compose(kf, kg)
        num = np.linalg.norm(P - Mmap[kfg]) ** 2
        den = np.linalg.norm(Mmap[kfg]) ** 2
        if affine:
            num += np.linalg.norm(q - bmap[kfg]) ** 2
            den += np.linalg.norm(bmap[kfg]) ** 2
        return math.sqrt(num / (den + EPS))

    rho = [resid(*t) for t in triples]
    keys = sorted(Mmap.keys())
    null = []
    for _ in range(n_null):
        kf, kg, kfg = triples[rng.integers(len(triples))]
        bad = keys[rng.integers(len(keys))]
        if bad != kfg:
            null.append(resid(kf, kg, bad))
    rho_m = float(np.mean(rho)) if rho else float("nan")
    null_m = float(np.mean(null)) if null else float("nan")
    return dict(rho=rho_m, rho_std=float(np.std(rho)) if rho else float("nan"),
                rho_null=null_m, rho_ratio=rho_m / (null_m + EPS),
                n_triples=len(triples))


# ------------------------------------------------------------------ verdict / io

def verdict(sep, e1, e2, p_true=None, N_u=None, floor=None, pr_cv=None):
    s = sep["sep_full"]
    dim = e1.get("id_twonn", float("nan"))
    if not np.isfinite(dim):
        dim = e1["pr"]
    dim = dim - (floor.get("id_twonn", 0.0) if floor else 0.0)
    rr = e2["rho_ratio"] if e2 else float("nan")

    factors = s < 0.25
    if p_true is not None:
        rich = 0.5 * p_true <= dim <= 3.0 * p_true
        tabular = N_u is not None and dim > 0.5 * N_u
    else:
        rich, tabular = dim > 1.5, False
    homo = rr < 0.5

    if pr_cv is not None and pr_cv > 0.15:
        v = "unstable_operating_point"
    elif not factors:
        v = "no_factorisation"
    elif tabular:
        v = "lookup_table"
    elif not rich:
        v = "ignores_code"
    elif homo:
        v = "eval_with_algebra"
    else:
        v = "eval_no_algebra"
    return dict(verdict=v, sep_full=s, dim=float(dim), pr=e1["pr"],
                rho_ratio=rr, pr_cv=pr_cv,
                factors=bool(factors), rich=bool(rich), homomorphic=bool(homo))


def jsonl_append(path, rec):
    if path is None:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
