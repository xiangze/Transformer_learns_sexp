"""
eval_common.py -- gauge-invariant probes for "does Phi factor as eval o ([[.]] (x) id)?"

Metrics only (numpy). Extraction of M(u,v) and A(u,v) from a live model lives in
eval_extract.py, which needs torch. Keeping them apart means the instrument
verification stage (S0) runs anywhere.

Notation (see design notes):
    Phi(u, v) = y            u = code of f (f-slot), v = code of x (x-slot)
    [[.]] : U -> [X,Y]       denotation map, unknown, learned
    eval  : [X,Y] (x) X -> Y counit, fixed by the universal property
    M(u,v) = d y / d v       estimator of [[u]] on the value path (SMCC side)
    A(u,v)                   attention matrices          (Markov side)

Three questions, three statistics:
    Sep    does Phi factor at all?          M must not depend on v
    r_op   how rich is the image of [[.]]?  effective rank of {M(u)}_u
    rho    is the image a representation?   M(f o g) =? M(f) M(g)
"""

import json
import math
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- Sep (factorisation)

def variance_decomposition(T):
    """Two-way ANOVA-style decomposition of a grid of observations.

    T : (N_u, N_v, ...) array. Trailing axes are the observation (a flattened
        matrix M, or a flattened stack of attention maps).

    Returns dict with, all in squared-Frobenius units per cell:
        V_u    variance explained by the f-slot main effect
        V_v    variance explained by the x-slot main effect
        V_int  interaction (the part explained by neither margin alone)
        V_tot  total
        sep_raw   V_v / V_u             (the ratio as literally defined)
        sep_full  (V_v + V_int) / V_u   (the one the verdict uses)

    Why sep_full is the operative one: a lookup table indexed by the *pair*
    (u, v) has no reason to show a large v main effect -- it shows a large
    INTERACTION. Var_v/Var_u alone would let a table pass. The factorisation
    Phi = eval o ([[.]] (x) id) requires M to be a function of u alone, i.e.
    both V_v and V_int must vanish relative to V_u.
    """
    T = np.asarray(T, dtype=np.float64)
    Nu, Nv = T.shape[0], T.shape[1]
    X = T.reshape(Nu, Nv, -1)

    grand = X.mean(axis=(0, 1), keepdims=True)          # (1,1,m)
    mu_u = X.mean(axis=1, keepdims=True)                # (Nu,1,m)
    mu_v = X.mean(axis=0, keepdims=True)                # (1,Nv,m)

    V_u = float(((mu_u - grand) ** 2).sum() / Nu)
    V_v = float(((mu_v - grand) ** 2).sum() / Nv)
    resid = X - mu_u - mu_v + grand
    V_int = float((resid ** 2).sum() / (Nu * Nv))
    V_tot = float(((X - grand) ** 2).sum() / (Nu * Nv))

    return dict(
        V_u=V_u, V_v=V_v, V_int=V_int, V_tot=V_tot,
        sep_raw=V_v / (V_u + EPS),
        sep_full=(V_v + V_int) / (V_u + EPS),
        frac_u=V_u / (V_tot + EPS),
        frac_v=V_v / (V_tot + EPS),
        frac_int=V_int / (V_tot + EPS),
    )


# ---------------------------------------------------------------- E1 (operator manifold)

def intrinsic_dim_twonn(M_u, discard=0.1):
    """TwoNN intrinsic dimension (Facco et al. 2017) of the point cloud {M(u)}.

    This exists because `effective_rank` measures LINEAR dimension, which is an
    upper bound on the manifold dimension and a loose one: a p-parameter family
    of operators traced over a wide parameter range spans far more than p linear
    directions (exp(theta.G) is not affine in theta). Claiming r_op ~ p from the
    SVD alone is only valid in the small-perturbation regime. Report both; if
    they disagree, the SVD number is the artefact.
    """
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
    d = float((x @ y) / (x @ x + EPS))          # regression through the origin
    return dict(id_twonn=d, n=N)


def effective_rank(M_u, center=True, energy=0.99):
    """Effective dimension of the image of [[.]].

    M_u : (N_u, ...) stack of operators, one per code u (already averaged over v).

    Returns:
        pr        participation ratio (sum s^2)^2 / sum s^4 -- smooth, no threshold
        r_energy  smallest k with cumulative spectral energy >= `energy`
        svals     normalised singular values (for plotting / noise-floor subtraction)

    The prediction that makes this worth measuring: if u ranges over a
    CONTINUOUS p-parameter family of functions and the model implements a
    genuine operator family, r_op ~ p. A lookup table gives r_op ~ N_u; a model
    that ignores u gives r_op ~ 0. Discrete function sets destroy the
    distinction -- the family must be continuously parameterised.
    """
    X = np.asarray(M_u, dtype=np.float64).reshape(len(M_u), -1)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    tot = s2.sum()
    if tot < EPS:
        return dict(pr=0.0, r_energy=0, svals=s.tolist(), total_energy=float(tot))
    pr = float(tot ** 2 / (s2 ** 2).sum())
    cum = np.cumsum(s2) / tot
    r_energy = int(np.searchsorted(cum, energy) + 1)
    return dict(pr=pr, r_energy=r_energy,
                svals=(s / (s[0] + EPS)).tolist(),
                total_energy=float(tot))


# ---------------------------------------------------------------- E2 (functoriality)

def functoriality_residual(Mmap, bmap, triples, n_null=200, rng=None, markov=False):
    """Is u -> [[u]] a homomorphism of the program algebra?

    Mmap[key]   (k,k) operator for code `key`
    bmap[key]   (k,)  affine offset, or None for the linear model
    triples     list of (key_f, key_g, key_fg) with fg = f o g in the language

    Affine composition, following the LRE parameterisation y ~ M v + b:
        (M_f, b_f) o (M_g, b_g) = (M_f M_g,  M_f b_g + b_f)

    Returns rho (mean relative residual) and rho_null, the same quantity on
    mismatched pairs. rho alone is meaningless: if all M(u) are similar,
    the product is trivially close. The verdict uses rho / rho_null.

    markov=True renormalises the predicted product to a row-stochastic matrix,
    for the attention-side variant where A lives in the Kleisli category of the
    distribution monad and composition must stay in it.
    """
    rng = rng or np.random.default_rng(0)
    affine = bmap is not None

    def compose(kf, kg):
        P = Mmap[kf] @ Mmap[kg]
        if markov:
            P = np.clip(P, 0, None)
            P = P / (P.sum(axis=-1, keepdims=True) + EPS)
        if not affine:
            return P, None
        return P, Mmap[kf] @ bmap[kg] + bmap[kf]

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
        kfg_bad = keys[rng.integers(len(keys))]
        if kfg_bad == kfg:
            continue
        null.append(resid(kf, kg, kfg_bad))

    rho_m = float(np.mean(rho))
    null_m = float(np.mean(null)) if null else float("nan")
    return dict(rho=rho_m, rho_std=float(np.std(rho)),
                rho_null=null_m,
                rho_ratio=rho_m / (null_m + EPS),
                n_triples=len(triples))


# ---------------------------------------------------------------- gauge checks

def gauge_report(stats_ref, stats_gauged, tol=0.05):
    """Stage-0 guard. Any statistic that moves under a function-preserving
    reparameterisation is disqualified regardless of how cleanly it separates
    conditions. Reports relative drift per key."""
    out = {}
    for k, v in stats_ref.items():
        if not isinstance(v, (int, float)) or k not in stats_gauged:
            continue
        w = stats_gauged[k]
        d = abs(w - v) / (abs(v) + EPS)
        out[k] = dict(ref=v, gauged=w, rel_drift=d, invariant=bool(d < tol))
    out["all_invariant"] = all(x["invariant"] for x in out.values()
                               if isinstance(x, dict))
    return out


# ---------------------------------------------------------------- verdict

def verdict(sep, e1, e2, p_true=None, N_u=None, floor=None):
    """Map the three statistics onto the decision table.

    floor: optional dict of the same statistics measured on a random-init model,
    used to reject r_op values that are pure estimation noise.
    """
    s = sep["sep_full"]
    pr = e1["pr"]
    rr = e2["rho_ratio"] if e2 else float("nan")

    pr_floor = floor["pr"] if floor else 0.0
    pr_net = pr - pr_floor

    factors = s < 0.25
    if p_true is not None:
        rich = (pr_net <= 3.0 * p_true) and (pr_net >= 0.5 * p_true)
        tabular = N_u is not None and pr_net > 0.5 * N_u
    else:
        rich, tabular = pr_net > 1.5, False
    homo = rr < 0.5

    if not factors:
        v = "no_factorisation"          # Phi does not split; neither reading holds
    elif tabular:
        v = "lookup_table"              # factors, but image is one-hot in u
    elif not rich:
        v = "ignores_code"              # image is (near) a point: u unused
    elif homo:
        v = "eval_with_algebra"         # the target result
    else:
        v = "eval_no_algebra"           # operator family, but not a representation
    return dict(verdict=v, sep_full=s, pr=pr, pr_net=pr_net, rho_ratio=rr,
                factors=bool(factors), rich=bool(rich), homomorphic=bool(homo))


# ---------------------------------------------------------------- io

def jsonl_append(path, rec):
    if path is None:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def jsonl_done(path):
    """Resumable runs: set of already-completed condition keys."""
    done = set()
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line).get("key"))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return done
