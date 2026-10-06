"""probe_mlp.py -- gauge/fiber-invariant eval probes, with explicit MLP handling.

Quantities
----------
M(u, v)      dF/dv, an N_DIM x N_DIM Jacobian.  Fiber-invariant by construction
             (depends only on the realized function F), covariant as M -> P M S^-1
             under readout / input reparameterization.

T(u,u'; v)   M(u',v)^+ M(u,v).  spec T is invariant under P, S in GL when the
             column spaces agree; we whiten first so the residual freedom is O.

Sep          two-way variance decomposition of vec M(u,v) over the (u, v) grid.
             eta^2_u = between-u variance / total.  Under linear-lambda eval this
             is exactly 1; the shortfall measures MLP nonlinearity leaking into
             the v-path.  (Rule 2 in the notes: this null hypothesis is only
             well posed because the probe language is linear in v.)

sep_spec     std over v of the sorted eigenvalue angles of T.  The GL-invariant
             form of Sep -- use this one when P, S are not orthogonal.

rho          functoriality residual.  Linear form  ||M(uu') - M(u)M(u')|| / ||M(uu')||
             and the GL-invariant spectral form  |spec(M(uu')^-1 M(u) M(u')) - 1|.
             rho_nl is the chain-rule form for nonlinear languages.

id_op        TwoNN intrinsic dimension of {vec M(u)}.  Ground truth for this
             language is dim O(3) = 3, NOT the number of programs.

ffn_eta      Rule 4: eta^2_u of the MLP neuron-coefficient vector (Geva key-value
             readout).  Permutation-gauge-invariant but NOT fiber-invariant, so it
             is reported as a LOCALIZATION signal only, never as eval evidence.

Path decomposition (rule 3 / localization)
------------------------------------------
--paths full,noMLP,noA,val   freezes MLP outputs and/or attention probabilities
at a reference v0, removing those v-paths.  Compare Sep and rho across modes.

Usage
-----
    python probe_mlp.py --ckpt ckpt_eval.pt --n-u 64 --n-v 16
    python probe_mlp.py --ckpt ckpt_eval.pt --mlp-sweep
    python probe_mlp.py --oracle          # stage-0 validation, no model needed
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from model_num import (PATH_FREEZE, ModelCfg, NumTransformer, PathCtl,
                       capture_reference, ctl_for)
from sexp_cont import (GRID, LANG, N_DIM, compose, encode_with_v, n_programs,
                       num_positions, out_positions, program_matrix,
                       sample_composable_pair, sample_program)

# ==========================================================================
# invariant statistics (numpy only -- mirrors eval_common.py conventions)
# ==========================================================================
def whiten(M: np.ndarray, Wy: np.ndarray, Wx: np.ndarray) -> np.ndarray:
    return Wy @ M @ Wx


def whiteners(Ms: np.ndarray, eps: float = 1e-8):
    """Ms: [N, m, n].  Returns (Wy, Wx) so that the GL freedom in M -> P M S^-1
    collapses to an orthogonal one after whitening."""
    m, n = Ms.shape[1], Ms.shape[2]
    Cy = np.einsum("kij,klj->il", Ms, Ms) / Ms.shape[0]
    Cx = np.einsum("kji,kjl->il", Ms, Ms) / Ms.shape[0]
    def isqrt(C, d):
        w, V = np.linalg.eigh(C + eps * np.eye(d))
        return V @ np.diag(w ** -0.5) @ V.T
    return isqrt(Cy, m), isqrt(Cx, n)


def spec_T(Ma: np.ndarray, Mb: np.ndarray) -> np.ndarray:
    """Eigenvalues of T = Mb^+ Ma, sorted by angle then modulus."""
    T, *_ = np.linalg.lstsq(Mb, Ma, rcond=None)
    ev = np.linalg.eigvals(T)
    order = np.lexsort((np.abs(ev), np.angle(ev)))
    return ev[order]


def sep_eta2(M: np.ndarray) -> float:
    """M: [n_u, n_v, m, n].  eta^2_u of vec M under two-way decomposition."""
    X = M.reshape(M.shape[0], M.shape[1], -1)
    gm = X.mean(axis=(0, 1), keepdims=True)
    um = X.mean(axis=1, keepdims=True)
    ss_tot = ((X - gm) ** 2).sum()
    ss_u = (((um - gm) ** 2) * X.shape[1]).sum()
    return float(ss_u / (ss_tot + 1e-12))


def sep_spec(M: np.ndarray, pairs: int = 64, seed: int = 0) -> float:
    """Mean over (u,u') pairs of the std over v of spec T angles.  0 under eval."""
    rng = np.random.default_rng(seed)
    n_u, n_v = M.shape[0], M.shape[1]
    if n_u < 2 or n_v < 2:
        return float("nan")
    acc = []
    for _ in range(pairs):
        i, j = rng.choice(n_u, size=2, replace=False)
        angs = np.stack([np.angle(spec_T(M[i, k], M[j, k])) for k in range(n_v)])
        acc.append(np.std(np.unwrap(angs, axis=0), axis=0).mean())
    return float(np.mean(acc))


def charpoly_inv(T: np.ndarray) -> np.ndarray:
    """Characteristic-polynomial coefficients of T via Newton's identities.

    Conjugation-invariant (so GL-invariant after whitening) and ordering-free,
    unlike a sorted eigenvalue list.  This is what makes sep_inv usable in the
    converged regime; see sep_inv.
    """
    n = T.shape[0]
    Tk = np.eye(n)
    traces = []
    for _ in range(n):
        Tk = Tk @ T
        traces.append(np.trace(Tk))
    e = [1.0]
    for k in range(1, n + 1):
        s = 0.0
        for i in range(1, k + 1):
            s += (-1) ** (i - 1) * e[k - i] * traces[i - 1]
        e.append(s / k)
    return np.array(e[1:])


def sep_inv(M: np.ndarray, pairs: int = 64, seed: int = 0) -> float:
    """GL-invariant v-dependence of T(u,u'), with no eigenvalue ordering.

    PREFER THIS OVER Sep AND sep_spec ONCE THE MODEL HAS CONVERGED.  Sep
    saturates at 1 for small error and sep_spec has a floor (~0.12) caused by
    sort-order flips between near-degenerate eigenvalues; sep_inv stays linear
    in the leakage amplitude over at least three decades (sep_inv ~ 0.75 eps_v).
    """
    rng = np.random.default_rng(seed)
    n_u, n_v = M.shape[0], M.shape[1]
    if n_u < 2 or n_v < 2:
        return float("nan")
    acc = []
    for _ in range(pairs):
        i, j = rng.choice(n_u, size=2, replace=False)
        C = []
        for k in range(n_v):
            T, *_ = np.linalg.lstsq(M[j, k], M[i, k], rcond=None)
            C.append(charpoly_inv(T))
        C = np.stack(C)
        acc.append(np.mean(C.std(0) / (np.abs(C.mean(0)) + 1.0)))
    return float(np.mean(acc))


def sample_v(rng, n_v: int, dist: str = "gauss") -> np.ndarray:
    """Probe points for v.  MUST match the model's training support.

    A grid-trained (lookup) model probed at gauss points is evaluated off
    support: between grid nodes it interpolates, so dF/dv picks up the
    interpolation wobble and sep_inv fires for a reason that has nothing to do
    with whether the model is a lookup table.  Report the scale-matched number
    too before attributing a high sep_inv to lookup structure.
    """
    if dist == "gauss":
        return rng.normal(size=(n_v, N_DIM))
    if dist == "grid":
        return rng.choice(GRID, size=(n_v, N_DIM))
    if dist == "gridjit":
        return rng.choice(GRID, size=(n_v, N_DIM)) + 0.02 * rng.normal(size=(n_v, N_DIM))
    raise ValueError(dist)


def null_expectations(eps: float) -> dict:
    """What a GENUINELY functorial model with Jacobian error `eps` should show.

    A nonzero rho is not evidence against functoriality until it exceeds this.
    First order: M(uu') - M(u)M(u') = eps(E_uu' - E_u A_u' - A_u E_u') + O(eps^2),
    so rho_lin ~ sqrt(3) eps for orthogonal A and independent errors.  The
    sep_inv slope is measured by simulation (sep_fix.py).
    """
    return {"rho_lin_null": np.sqrt(3) * eps,
            "rho_spec_null": 1.19 * eps,
            "sep_inv_null": 0.75 * eps,
            "Sep_null": 1.0 - eps ** 2}


def rho_linear(M_uu: np.ndarray, M_u: np.ndarray, M_u2: np.ndarray) -> float:
    d = M_uu - M_u @ M_u2
    return float(np.linalg.norm(d) / (np.linalg.norm(M_uu) + 1e-12))


def rho_spectral(M_uu: np.ndarray, M_u: np.ndarray, M_u2: np.ndarray) -> float:
    """GL-invariant: spec(M_uu^-1 M_u M_u') should be {1,...,1}."""
    try:
        C = np.linalg.solve(M_uu, M_u @ M_u2)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.abs(np.linalg.eigvals(C) - 1.0).mean())


def twonn(X: np.ndarray, discard: float = 0.1) -> float:
    """TwoNN intrinsic dimension (Facco et al.).  X: [N, D]."""
    N = X.shape[0]
    if N < 10:
        return float("nan")
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, np.inf)
    d = np.sqrt(np.sort(D2, axis=1)[:, :2])
    ok = d[:, 0] > 1e-12
    mu = np.sort(d[ok, 1] / d[ok, 0])
    n = len(mu)
    keep = int(n * (1 - discard))
    mu, Fe = mu[:keep], np.arange(1, keep + 1) / n
    x, y = np.log(mu), -np.log(1 - Fe)
    return float((x @ y) / (x @ x + 1e-12))


# ==========================================================================
# stage 0: oracle validation (no model)
# ==========================================================================
def oracle(n_u=48, n_v=12, seed=0):
    """Four synthetic operator families with known verdicts."""
    rng = np.random.default_rng(seed)
    progs = [sample_program(rng) for _ in range(n_u)]
    vs = rng.normal(size=(n_v, N_DIM))
    A = np.stack([program_matrix(u) for u in progs])           # [n_u, 3, 3]

    fams = {}
    # true eval: M depends on u only
    fams["true_eval"] = np.broadcast_to(A[:, None], (n_u, n_v, N_DIM, N_DIM)).copy()
    # table over f: per-(u,v) independent perturbation (a lookup table)
    fams["table_over_f"] = A[:, None] + 0.6 * rng.normal(size=(n_u, n_v, N_DIM, N_DIM))
    # lookup_pair: depends on v as much as on u
    B = rng.normal(size=(n_v, N_DIM, N_DIM))
    fams["lookup_pair"] = 0.5 * A[:, None] + 0.5 * B[None, :]
    # ignores code
    fams["ignores_code"] = np.broadcast_to(A[0][None, None], (n_u, n_v, N_DIM, N_DIM)).copy()

    rows = []
    for name, M in fams.items():
        Wy, Wx = whiteners(M.reshape(-1, N_DIM, N_DIM))
        Mw = np.stack([[whiten(M[i, k], Wy, Wx) for k in range(n_v)] for i in range(n_u)])
        rows.append(dict(
            family=name,
            Sep=round(sep_eta2(Mw), 4),
            sep_spec=round(sep_spec(Mw, pairs=24, seed=seed), 4),
            id_op=round(twonn(M[:, 0].reshape(n_u, -1)), 3),
        ))
    return rows


# ==========================================================================
# model-side extraction
# ==========================================================================
def load_model(path, device):
    if not os.path.exists(path):
        hint = ""
        if "lookup" in path:
            hint = ("\nThis one is the memorizable lookup control; build it with\n"
                    "  SEXP_N_ANGLE=4 SEXP_MAX_DEPTH=2 python train_cont.py --mode grid \\\n"
                    "      --steps 50000 --bsz 512 --lr 1e-3 --save-every 5000 \\\n"
                    f"      --out {path}")
        near = sorted(glob.glob("*.pt") + glob.glob("*/*.pt"))
        raise SystemExit(
            f"no such checkpoint: {path}\n"
            f"checkpoints that do exist here: {near if near else '(none)'}"
            f"{hint}\nTrain it first (make_ckpts.sh builds the whole matched set), "
            "or point --ckpt at one of the above.")
    ck = torch.load(path, map_location=device, weights_only=False)
    saved = ck.get("lang")
    if saved is None:
        print(f"  WARNING: {path} has no language fingerprint (trained before the "
              f"SEXP_* override was added); assuming it matches the current one.")
    elif saved != LANG:
        diff = {k: (saved.get(k), LANG.get(k)) for k in set(saved) | set(LANG)
                if saved.get(k) != LANG.get(k)}
        raise SystemExit(
            f"language mismatch for {path}: (checkpoint, current) = {diff}\n"
            f"Re-run this probe with the same SEXP_* environment variables used "
            f"for training, e.g.  SEXP_N_ANGLE={saved.get('N_ANGLE')} "
            f"SEXP_MAX_DEPTH={saved.get('MAX_DEPTH')} python probe_mlp.py ...")
    model = NumTransformer(ModelCfg(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def jacobian_M(model, u, v, out_pos, num_pos, device, ctl=None):
    """dF/dv at (u, v).  Returns [N_DIM, N_DIM] numpy."""
    ids_np, vals_np = encode_with_v(u, v)
    ids = torch.from_numpy(ids_np).unsqueeze(0).to(device)
    vfree = torch.tensor(np.asarray(v, dtype=np.float32), device=device,
                         requires_grad=True)
    vals = torch.zeros(1, ids.shape[1], device=device)
    vals = vals.index_put((torch.zeros(N_DIM, dtype=torch.long, device=device),
                           torch.from_numpy(num_pos).to(device)), vfree)
    y, _ = model(ids, vals, out_pos, ctl)
    rows = []
    for k in range(N_DIM):
        g, = torch.autograd.grad(y[0, k], vfree, retain_graph=(k < N_DIM - 1))
        rows.append(g.detach().cpu().numpy())
    return np.stack(rows)


def ffn_coeffs(model, u, v, out_pos, num_pos, device):
    """Geva key-value neuron coefficients at the last <OUT> position, all layers."""
    ids_np, vals_np = encode_with_v(u, v)
    ids = torch.from_numpy(ids_np).unsqueeze(0).to(device)
    vals = torch.from_numpy(vals_np).unsqueeze(0).to(device)
    with torch.no_grad():
        _, cache = model(ids, vals, out_pos, PathCtl(capture=True))
    pos = int(out_pos[-1].item())
    return np.concatenate([cache["ffn_act"][l][0, pos].cpu().numpy()
                           for l in sorted(cache["ffn_act"])])


def extract_grid(model, progs, vs, out_pos, num_pos, device, path="full",
                 ablate=None, ref_v=None):
    """M[u, v] under a given path-freezing / ablation condition."""
    n_layer = len(model.blocks)
    ctl_base = PathCtl()
    if ablate:
        # mean MLP output over the probe distribution, per layer
        accum = {l: [] for l in ablate}
        for u in progs[: min(len(progs), 16)]:
            ids_np, vals_np = encode_with_v(u, vs[0])
            ids = torch.from_numpy(ids_np).unsqueeze(0).to(device)
            vals = torch.from_numpy(vals_np).unsqueeze(0).to(device)
            with torch.no_grad():
                _, c = model(ids, vals, out_pos, PathCtl(capture=True))
            for l in ablate:
                accum[l].append(c["mlp_out"][l][0])
        ctl_base.ablate_mlp = set(ablate)
        ctl_base.mean_mlp = {l: torch.stack(accum[l]).mean(0) for l in ablate}

    ref_v = vs[0] if ref_v is None else ref_v
    M = np.zeros((len(progs), len(vs), N_DIM, N_DIM))
    for i, u in enumerate(progs):
        if path != "full":
            ids_np, vals_np = encode_with_v(u, ref_v)
            ids = torch.from_numpy(ids_np).unsqueeze(0).to(device)
            vals = torch.from_numpy(vals_np).unsqueeze(0).to(device)
            ref = capture_reference(model, ids, vals, out_pos)
            ctl = ctl_for(path, ref, n_layer)
            ctl.ablate_mlp, ctl.mean_mlp = ctl_base.ablate_mlp, ctl_base.mean_mlp
        else:
            ctl = ctl_base
        for k, v in enumerate(vs):
            M[i, k] = jacobian_M(model, u, v, out_pos, num_pos, device, ctl)
    return M


def _ctl_per_program(model, u, ref_v, out_pos, device, path):
    """Freeze-control captured from THIS program's own reference forward pass.

    Freezing at another program's reference would delete the u-path as well as
    the v-path, which is not what the 2x2 decomposition is meant to test.
    """
    if path == "full":
        return None
    ids_np, vals_np = encode_with_v(u, ref_v)
    ref = capture_reference(model,
                            torch.from_numpy(ids_np).unsqueeze(0).to(device),
                            torch.from_numpy(vals_np).unsqueeze(0).to(device),
                            out_pos)
    return ctl_for(path, ref, len(model.blocks))


def functoriality(model, out_pos, num_pos, device, n_pairs, v0, seed, path="full"):
    rng = np.random.default_rng(seed)
    lin, spec, lin_gt = [], [], []
    for _ in range(n_pairs):
        a, b = sample_composable_pair(rng)
        ab = compose(a, b)
        Ma = jacobian_M(model, a, v0, out_pos, num_pos, device,
                        _ctl_per_program(model, a, v0, out_pos, device, path))
        Mb = jacobian_M(model, b, v0, out_pos, num_pos, device,
                        _ctl_per_program(model, b, v0, out_pos, device, path))
        Mab = jacobian_M(model, ab, v0, out_pos, num_pos, device,
                         _ctl_per_program(model, ab, v0, out_pos, device, path))
        lin.append(rho_linear(Mab, Ma, Mb))
        spec.append(rho_spectral(Mab, Ma, Mb))
        lin_gt.append(rho_linear(program_matrix(ab),
                                 program_matrix(a), program_matrix(b)))
    spec_ok = [x for x in spec if np.isfinite(x)]
    return (float(np.mean(lin)),
            float(np.mean(spec_ok)) if spec_ok else float('nan'),
            float(np.mean(lin_gt)))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--n-u", type=int, default=48)
    ap.add_argument("--n-v", type=int, default=12)
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--paths", default="full,noA,noV,noMLP,frozen",
                    help="comma-separated path conditions. 'noA'/'noV'/'noMLP' each "
                         "remove exactly ONE v-path, so their sep_excess values are "
                         "directly comparable; 'val'/'patOnly'/'mlpOnly' leave exactly "
                         "one path live; 'frozen' freezes all three and must give "
                         "dF/dv ~ 0 -- it is the validation that freezing does not "
                         f"leak. Available: {sorted(PATH_FREEZE)}")
    ap.add_argument("--mlp-sweep", action="store_true")
    ap.add_argument("--v-dist", default="gauss",
                    choices=["gauss", "grid", "gridjit"],
                    help="where the probe draws v from. 'gauss' = N(0,1), which is "
                         "the --mode eval training distribution. 'grid' = exactly "
                         "the GRID points used by --mode grid. 'gridjit' = grid "
                         "points plus small jitter, so the Jacobian is taken near "
                         "but not on the training support. A grid-trained model "
                         "probed with 'gauss' is being probed OFF SUPPORT, which "
                         "inflates both |M-GT| and sep_inv; always report the "
                         "scale-matched number alongside it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.oracle:
        rows = oracle(seed=args.seed)
        print("--- stage 0 oracle ---")
        for r in rows:
            print(f"  {r['family']:14s} Sep={r['Sep']:.4f}  "
                  f"sep_spec={r['sep_spec']:.4f}  id_op={r['id_op']:.2f}")
        if not args.ckpt:
            return

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    out_pos = torch.from_numpy(out_positions()).to(device)
    num_pos = num_positions()
    rng = np.random.default_rng(args.seed)
    progs = [sample_program(rng) for _ in range(args.n_u)]
    vs = sample_v(rng, args.n_v, args.v_dist)
    results_vdist = args.v_dist
    results = {"ckpt": args.ckpt, "trained_mode": ck.get("mode"),
               "v_dist": args.v_dist, "lang": ck.get("lang"), "paths": {}}

    print(f"\n--- {args.ckpt}  (trained mode: {ck.get('mode')}, steps: {ck.get('steps')}, "
          f"probed at v~{args.v_dist}) ---")
    if ck.get("mode") == "grid" and args.v_dist == "gauss":
        print("  WARNING: grid-trained model probed at gauss points -> OFF SUPPORT.\n"
              "  |M-GT| and sep_inv are both inflated by interpolation wobble, not\n"
              "  only by lookup structure.  Re-run with --v-dist grid (and gridjit)\n"
              "  and compare before attributing a high sep_inv to lookup.")
    if ck.get("mode") in ("eval", "ignore_code") and args.v_dist != "gauss":
        print(f"  NOTE: {ck.get('mode')}-trained model probed at {args.v_dist}; its\n"
              "  training support was gauss, so this is the scale-matched comparison\n"
              "  run, not the primary one.")
    for path in args.paths.split(","):
        M = extract_grid(model, progs, vs, out_pos, num_pos, device, path=path)
        Wy, Wx = whiteners(M.reshape(-1, N_DIM, N_DIM))
        Mw = np.stack([[whiten(M[i, k], Wy, Wx) for k in range(len(vs))]
                       for i in range(len(progs))])
        rl, rs, rgt = functoriality(model, out_pos, num_pos, device,
                                    args.n_pairs, vs[0], args.seed, path)
        mnorm = float(np.mean([np.linalg.norm(M[i, 0]) for i in range(len(progs))]))
        if path == args.paths.split(",")[0]:
            norm_ref = mnorm
        relnorm = mnorm / (norm_ref + 1e-12)
        fit = float(np.mean([np.linalg.norm(M[i, 0] - program_matrix(progs[i]))
                             / np.linalg.norm(program_matrix(progs[i]))
                             for i in range(len(progs))]))
        null = null_expectations(fit)
        row = dict(
            Sep=round(sep_eta2(Mw), 5),
            sep_spec=round(sep_spec(Mw, pairs=24, seed=args.seed), 4),
            sep_inv=round(sep_inv(Mw, pairs=24, seed=args.seed), 5),
            id_op=round(twonn(M[:, 0].reshape(len(progs), -1)), 3),
            rho_lin=round(rl, 4),
            rho_spec=round(rs, 4),
            rho_oracle=round(rgt, 6),
            fit=round(fit, 5),
            relnorm=round(relnorm, 4),
            **{k: round(float(v), 5) for k, v in null.items()},
        )
        # excess over the functorial-with-noise null: >1 means genuinely non-functorial
        row["rho_excess"] = round(rl / (null["rho_lin_null"] + 1e-12), 2)
        row["sep_excess"] = round(row["sep_inv"] / (null["sep_inv_null"] + 1e-12), 2)
        results["paths"][path] = row
        flag = ""
        if path == "frozen":
            flag = "   <- VALIDATION: all paths frozen, must be ~0"
        elif relnorm < 0.15:
            flag = (f"   <- NO SIGNAL (relnorm={relnorm:.3f}): this condition "
                    "removed the operator, so sep_inv/rho here are noise ratios, "
                    "not v-dependence")
        print(f"  path={path:7s} relnorm={relnorm:.3f} |M-GT|={row['fit']:.5f}  "
              f"sep_inv={row['sep_inv']:.5f} (x{row['sep_excess']})  "
              f"rho_lin={row['rho_lin']:.4f} (x{row['rho_excess']})  "
              f"id_op={row['id_op']:5.2f}{flag}")

    # rule 4: MLP neuron coefficients -- localization only, not evidence
    C = np.stack([[ffn_coeffs(model, u, v, out_pos, num_pos, device) for v in vs]
                  for u in progs])
    results["ffn_eta2_u"] = round(sep_eta2(C[..., None, :]), 4)
    print(f"  [localization only] FFN neuron-coeff eta^2_u = {results['ffn_eta2_u']:.4f}"
          "   (permutation-gauge-invariant, NOT fiber-invariant)")

    if args.mlp_sweep:
        print("  --- MLP mean-ablation sweep ---")
        results["mlp_sweep"] = {}
        for l in range(len(model.blocks)):
            M = extract_grid(model, progs, vs, out_pos, num_pos, device,
                             path="full", ablate=[l])
            Wy, Wx = whiteners(M.reshape(-1, N_DIM, N_DIM))
            Mw = np.stack([[whiten(M[i, k], Wy, Wx) for k in range(len(vs))]
                           for i in range(len(progs))])
            fit = float(np.mean([np.linalg.norm(M[i, 0] - program_matrix(progs[i]))
                                 / np.linalg.norm(program_matrix(progs[i]))
                                 for i in range(len(progs))]))
            results["mlp_sweep"][l] = dict(Sep=round(sep_eta2(Mw), 4),
                                           sep_inv=round(sep_inv(Mw, pairs=24,
                                                                 seed=args.seed), 5),
                                           fit=round(fit, 4))
            print(f"    ablate MLP[{l}]  Sep={sep_eta2(Mw):.4f}  |M-GT|={fit:.4f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
