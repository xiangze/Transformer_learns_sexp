"""
eval_layers.py -- where [[u]] lives: the (l_in, l_out) scan, position transfer,
and function-vector portability.

The scan replaces the earlier one-dimensional readout sweep. LRE selects the
injection layer per relation by grid search and its optima spread from layer 1
to 11, so holding l_in at the embedding was a hole: a model can look like it
never forms an operator simply because the probe was injecting in the wrong
place. Sweeping both ends gives a matrix whose entries are read as:

    below the diagonal   undefined (l_out < l_in)
    on the diagonal      also undefined: no block has run between injection and
                         readout, so with read_pos != x_pos nothing can reach
                         the readout and M is identically zero
    above the diagonal   the operator the model builds between the two depths

Off-diagonal cells are the only measurements. The signature of an apply depth
is a BAND, not a point: the row where Sep drops
tells you where the argument stops being needed as a token, the column where
r_op saturates tells you where the operator has finished forming. LRE also
predicts a rise-then-collapse along l_out rather than a monotone knee, because
late layers switch from enriching the subject to predicting the next token; the
diagonal control is what separates that mode switch from a genuine failure.
"""

import json

import numpy as np

from eval_common import (variance_decomposition, effective_rank,
                         intrinsic_dim_twonn, functoriality_residual,
                         verdict, EPS)


# ------------------------------------------------------------------ 2-D scan

def scan2d(probe, prompts, x_pos, read_pos, ep, layers_in=None, layers_out=None,
           triples=None, p_true=None, beta=1.0, progress=None,
           cache=None, cells=None):
    """prompts: list over u of list over v of `ids`.

    cache: path to a jsonl of per-cell results. Cells already present are
    skipped and reloaded, so a scan over a real checkpoint survives a killed job
    or an OOM and resumes where it stopped. Each line is one cell -- the unit of
    work is a cell, not the whole scan, because on a 7B model one cell is minutes.

    cells: explicit list of (l_in, l_out) to compute, for sharding across GPUs.
    Every rank writes to the same cache path pattern and a final pass with
    cells=None (or any rank) reassembles the matrix from the cache.

    Returns dict of (n_in, n_out) arrays plus the per-cell verdicts.
    """
    L = probe.n_layers
    layers_in = list(layers_in if layers_in is not None else range(0, L))
    layers_out = list(layers_out if layers_out is not None else range(1, L + 1))
    missing = [l for l in layers_in if l not in probe.B]
    if missing:
        raise ValueError(f"no injection frame for layers {missing}: build them "
                         f"with build_in_basis and pass them to TinyProbe")

    shape = (len(layers_in), len(layers_out))
    keys = ("sep_full", "frac_u", "pr", "id_twonn", "rho_ratio", "spread")
    out = {k: np.full(shape, np.nan) for k in keys}
    verdicts = [[None] * shape[1] for _ in range(shape[0])]

    done = {}
    if cache:
        try:
            with open(cache) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                        done[(r["l_in"], r["l_out"])] = r
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

    want = set(cells) if cells is not None else None

    for i, li in enumerate(layers_in):
        for j, lo in enumerate(layers_out):
            if lo <= li:
                continue          # M is identically zero; see the module docstring
            if (li, lo) in done:
                r = done[(li, lo)]
                for k in keys:
                    out[k][i, j] = r.get(k, np.nan)
                verdicts[i][j] = r.get("verdict")
                continue
            if want is not None and (li, lo) not in want:
                continue
            M = np.zeros((len(prompts), len(prompts[0]), probe.k, probe.k))
            b = np.zeros((len(prompts), len(prompts[0]), probe.k))
            sp = []
            for iu, row in enumerate(prompts):
                for iv, ids in enumerate(row):
                    m, bb, s = probe.M(ids, x_pos, read_pos, li, lo, ep, v=iv,
                                       return_spread=True)
                    M[iu, iv], b[iu, iv] = m, bb
                    sp.append(s)
            sep = variance_decomposition(M)
            Mu = M.mean(axis=1)
            e1 = effective_rank(Mu)
            e1.update(intrinsic_dim_twonn(Mu))
            e2 = None
            if triples:
                e2 = functoriality_residual(
                    {n: Mu[n] for n in range(len(Mu))},
                    {n: b.mean(axis=1)[n] for n in range(len(Mu))},
                    triples, beta=beta)
            out["sep_full"][i, j] = sep["sep_full"]
            out["frac_u"][i, j] = sep["frac_u"]
            out["pr"][i, j] = e1["pr"]
            out["id_twonn"][i, j] = e1["id_twonn"]
            out["rho_ratio"][i, j] = (e2 or {}).get("rho_ratio", np.nan)
            out["spread"][i, j] = float(np.mean(sp))
            verdicts[i][j] = verdict(sep, e1, e2, p_true=p_true,
                                     N_u=len(prompts))["verdict"]
            if cache:
                rec = dict(l_in=li, l_out=lo, verdict=verdicts[i][j],
                           **{k: float(out[k][i, j]) for k in keys})
                with open(cache, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if progress:
                progress(li, lo)

    out["layers_in"], out["layers_out"] = layers_in, layers_out
    out["verdicts"] = verdicts
    return out


def format_scan(res, key="sep_full", fmt="{:7.2f}"):
    li, lo = res["layers_in"], res["layers_out"]
    lines = [f"  {key}", "  l_in\\l_out " + "".join(f"{l:>8}" for l in lo)]
    for i, a in enumerate(li):
        row = "".join("       ." if np.isnan(v) else fmt.format(v)
                      for v in res[key][i])
        lines.append(f"  {a:>10} " + row)
    return "\n".join(lines)


def apply_band(res, sep_thresh=0.25, dim_lo=None, dim_hi=None):
    """The (l_in, l_out) cells where the operator has both separated from the
    argument and reached a plausible dimension. Empty = no apply depth."""
    ok = []
    S, D = res["sep_full"], res["id_twonn"]
    for i, a in enumerate(res["layers_in"]):
        for j, c in enumerate(res["layers_out"]):
            if np.isnan(S[i, j]) or c == a:
                continue
            if S[i, j] >= sep_thresh:
                continue
            if dim_lo is not None and not (dim_lo <= D[i, j] <= dim_hi):
                continue
            ok.append((a, c, float(S[i, j]), float(D[i, j])))
    return ok


# ------------------------------------------------------------------ position

def position_transfer(probe, ids_of, x_positions, read_pos, l_in, l_out, ep,
                      codes_u):
    """Same code u, argument at different indices. ids_of(u, x_pos) -> ids."""
    Ms = {}
    for u in codes_u:
        for p in x_positions:
            Ms[(u, p)] = probe.M(ids_of(u, p), p, read_pos, l_in, l_out, ep, v=0)[0]
    same, diff = [], []
    for u in codes_u:
        for i, p in enumerate(x_positions):
            for q in x_positions[i + 1:]:
                a, b = Ms[(u, p)], Ms[(u, q)]
                same.append(np.linalg.norm(a - b) / (np.linalg.norm(a) + EPS))
    for p in x_positions:
        for i, u in enumerate(codes_u):
            for w in codes_u[i + 1:]:
                a, b = Ms[(u, p)], Ms[(w, p)]
                diff.append(np.linalg.norm(a - b) / (np.linalg.norm(a) + EPS))
    s, d = float(np.mean(same)), float(np.mean(diff))
    return dict(pos_resid=s, pos_null=d, pos_ratio=s / (d + EPS),
                position_independent=bool(s / (d + EPS) < 0.3))


# ------------------------------------------------------------------ FV portability

def extract_fv(probe, prompts_u, layer, heads=None, pos=-1):
    """theta(u) = mean over prompts of the summed contribution of selected heads
    at `pos`, as in Todd et al."""
    acc = None
    for ids in prompts_u:
        ho = probe.head_out(ids, layer, pos)
        sel = ho if heads is None else ho[heads]
        v = sel.sum(axis=0)
        acc = v if acc is None else acc + v
    return acc / len(prompts_u)


def fv_transfer_matrix(probe, prompts, blank_prompts, x_pos, read_pos, ep,
                       l_in, l_out, src_layers=None, dst_layers=None,
                       heads=None, scale=1.0):
    """gain[src, dst] = transfer - null.

    null (the same patch built from a different u) is mandatory: if the M(u) are
    all similar, transfer is trivially high and means nothing. A diagonal-only
    gain says the FV is a positional artefact of the layer it was read from; a
    broad off-diagonal plateau says u is a first-class value that survives
    relocation.
    """
    L = probe.n_layers
    src_layers = list(src_layers if src_layers is not None else range(1, L))
    dst_layers = list(dst_layers if dst_layers is not None else range(1, L))

    M_ref = {iu: probe.M(prompts[iu][0], x_pos, read_pos, l_in, l_out, ep, v=0)[0]
             for iu in range(len(prompts))}
    T = np.zeros((len(src_layers), len(dst_layers)))
    N = np.zeros_like(T)
    for si, ls in enumerate(src_layers):
        fvs = {iu: extract_fv(probe, prompts[iu], ls, heads)
               for iu in range(len(prompts))}
        for di, ld in enumerate(dst_layers):
            hit, null = [], []
            for iu in range(len(prompts)):
                ids = blank_prompts[iu][0]
                ref = M_ref[iu]
                Mp = probe.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0,
                             patch=(ld, read_pos, scale * fvs[iu]))[0]
                hit.append(1 - np.linalg.norm(Mp - ref) / (np.linalg.norm(ref) + EPS))
                jw = (iu + 1) % len(prompts)
                Mn = probe.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0,
                             patch=(ld, read_pos, scale * fvs[jw]))[0]
                null.append(1 - np.linalg.norm(Mn - ref) / (np.linalg.norm(ref) + EPS))
            T[si, di], N[si, di] = np.mean(hit), np.mean(null)
    g = T - N
    bi = int(np.argmax(g))
    return dict(src_layers=src_layers, dst_layers=dst_layers,
                transfer=T, null=N, gain=g,
                best=dict(src=src_layers[bi // len(dst_layers)],
                          dst=dst_layers[bi % len(dst_layers)],
                          gain=float(g.max())))
