"""
eval_layers.py -- where in the network does [[u]] exist, and how portable is it?

Three questions, all answered by recomputing M under a change of *where* we look
or *where* we inject.

  L1  LAYER SCAN.  M^(l)(u) = d(readout at layer l)/dc. Sweeping l gives a
      formation curve: Sep^(l) falls and r_op^(l) rises at the depth where the
      code stops being a token and starts being an operator. The knee is the
      operational definition of "apply depth". A model that never separates has
      no such knee.

  L2  POSITION TRANSFER.  Move the x-slot and the readout to different indices in
      a longer padded template. If [[u]] is a genuine closure, M(u) should be
      the same operator regardless of where the argument sits. Position
      dependence means the model learned a positional pattern, not a function.

  L3  FUNCTION-VECTOR PORTABILITY.  Extract the FV theta(u) the way Todd et al.
      do -- mean of selected attention-head outputs at the last position -- then
      inject it at a different layer and position in a prompt whose f-slot has
      been blanked, and recompute M. Two numbers come out:

        transfer[l_src, l_dst] = 1 - ||M_patched - M(u)|| / ||M(u)||
        null                   = the same with a mismatched u

      This is the sharp form of "is the FV layer/position independent": a
      closure that survives relocation is a first-class value; one that only
      works at the layer it was read from is a positional artefact. The null is
      mandatory -- if all M(u) are similar, transfer is trivially high.

Everything here is written against a small probe protocol so the same analysis
runs on the numpy reference model and on an HF checkpoint.
"""

import numpy as np

from eval_common import (variance_decomposition, effective_rank,
                         intrinsic_dim_twonn, functoriality_residual, EPS)


# ---------------------------------------------------------------- probe protocol

class Probe:
    """Implement these three and the whole module works.

      n_layers                       int
      M(ids, x_pos, read_pos, layer, c0=None, patch=None) -> (M, b)
      head_out(ids, layer, pos)      -> (n_heads, d) per-head contributions
    `patch` is (layer, pos, vector) added to the residual stream.
    """
    n_layers = 0

    def M(self, ids, x_pos, read_pos, layer, c0=None, patch=None):
        raise NotImplementedError

    def head_out(self, ids, layer, pos):
        raise NotImplementedError


class TinyProbe(Probe):
    """numpy reference implementation (see eval_gauge.TinyLM)."""

    def __init__(self, model, B_in, R_out, eps=1e-4):
        self.m, self.B, self.R, self.eps = model, B_in, R_out, eps
        self.n_layers = model.L

    def _fwd(self, ids, x_pos, c, patch):
        from eval_gauge import _rmsnorm, _softmax, _gelu
        m = self.m
        e = m.E[ids].copy()
        e[x_pos] = c @ self.B
        T = len(ids)
        h = e + m.P[:T]
        hs = [h]
        mask = np.triu(np.full((T, T), -1e9), 1)
        for l in range(m.L):
            if patch is not None and patch[0] == l:
                h = h.copy()
                h[patch[1]] = h[patch[1]] + patch[2]
            hn = _rmsnorm(h)
            outs = []
            for hd in range(m.H):
                q, k, v = hn @ m.Wq[l, hd], hn @ m.Wk[l, hd], hn @ m.Wv[l, hd]
                a = _softmax(q @ k.T / np.sqrt(m.dh) + mask)
                outs.append(a @ v)
            h = h + np.concatenate(outs, axis=-1) @ m.Wo[l]
            h = h + _gelu(_rmsnorm(h) @ m.W1[l]) @ m.W2[l]
            hs.append(h)
        return hs

    def M(self, ids, x_pos, read_pos, layer, c0=None, patch=None):
        from eval_gauge import _rmsnorm
        k = self.B.shape[0]
        c0 = np.zeros(k) if c0 is None else np.asarray(c0, float)

        def f(c):
            hs = self._fwd(ids, x_pos, c, patch)
            return self.R @ _rmsnorm(hs[layer])[read_pos]

        cols = []
        for i in range(k):
            cp, cm = c0.copy(), c0.copy()
            cp[i] += self.eps
            cm[i] -= self.eps
            cols.append((f(cp) - f(cm)) / (2 * self.eps))
        M = np.stack(cols, axis=1)
        return M, f(c0) - M @ c0

    def head_out(self, ids, layer, pos):
        from eval_gauge import _rmsnorm, _softmax
        m = self.m
        hs = self._fwd(ids, 0, np.zeros(self.B.shape[0]), None)
        h = hs[layer]
        hn = _rmsnorm(h)
        T = len(ids)
        mask = np.triu(np.full((T, T), -1e9), 1)
        outs = []
        for hd in range(m.H):
            q, k, v = hn @ m.Wq[layer, hd], hn @ m.Wk[layer, hd], hn @ m.Wv[layer, hd]
            a = _softmax(q @ k.T / np.sqrt(m.dh) + mask)
            z = np.zeros((T, m.d))
            z[:, hd * m.dh:(hd + 1) * m.dh] = a @ v
            outs.append((z @ m.Wo[layer])[pos])
        return np.stack(outs)


# ---------------------------------------------------------------- L1 layer scan

def layer_scan(probe, prompts, layers=None, triples=None, c0_of=None):
    """prompts: list over u of list over v of (ids, x_pos, read_pos).

    c0_of(v) -> expansion point in value coordinates. MUST be supplied and MUST
    depend on v: with a common c0 the x-slot embedding is identical for every v,
    Sep is vacuously 0, and the factorisation test passes for any model at all.
    This is the easiest way to get a meaningless pass out of this whole probe.

    Returns per-layer dict of sep_full / pr / id_twonn / rho_ratio."""
    layers = layers or list(range(1, probe.n_layers + 1))
    if c0_of is None:
        raise ValueError("c0_of is required: see the docstring. Pass a one-hot "
                         "at the argument token, or the probe is vacuous.")
    rows = []
    for l in layers:
        M = np.asarray([[probe.M(*p, layer=l, c0=c0_of(iv))[0]
                         for iv, p in enumerate(row)] for row in prompts])
        b = np.asarray([[probe.M(*p, layer=l, c0=c0_of(iv))[1]
                         for iv, p in enumerate(row)] for row in prompts])
        sep = variance_decomposition(M)
        Mu = M.mean(axis=1)
        e1 = effective_rank(Mu)
        e1.update(intrinsic_dim_twonn(Mu))
        e2 = None
        if triples:
            e2 = functoriality_residual({i: Mu[i] for i in range(len(Mu))},
                                        {i: b.mean(axis=1)[i] for i in range(len(Mu))},
                                        triples)
        rows.append(dict(layer=l, sep_full=sep["sep_full"], frac_u=sep["frac_u"],
                         pr=e1["pr"], id_twonn=e1["id_twonn"],
                         rho_ratio=(e2 or {}).get("rho_ratio", float("nan"))))
    return rows


def apply_depth(rows, sep_thresh=0.25):
    """Shallowest layer where the operator has separated. None = never."""
    for r in rows:
        if r["sep_full"] < sep_thresh:
            return r["layer"]
    return None


# ---------------------------------------------------------------- L2 position

def position_transfer(probe, ids_of, x_positions, read_pos, layer, codes_u):
    """Same code u, argument placed at different indices in a padded template.

    ids_of(u, x_pos) -> ids with the x-slot at that index and everything else
    identical. Returns mean relative disagreement between positions, and the
    null from comparing different u at the same position."""
    Ms = {}
    for u in codes_u:
        for p in x_positions:
            Ms[(u, p)] = probe.M(ids_of(u, p), p, read_pos, layer)[0]

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


# ---------------------------------------------------------------- L3 FV portability

def extract_fv(probe, prompts_u, layer, heads=None, pos=-1):
    """theta(u) = mean over prompts of the summed contribution of selected heads
    at `pos`, following Todd et al. heads=None uses all heads."""
    acc = None
    for (ids, x_pos, read_pos) in prompts_u:
        ho = probe.head_out(ids, layer, pos % len(ids))
        sel = ho if heads is None else ho[heads]
        v = sel.sum(axis=0)
        acc = v if acc is None else acc + v
    return acc / len(prompts_u)


def fv_transfer_matrix(probe, prompts, blank_prompts, codes_u, read_pos,
                       readout_layer=None, src_layers=None, dst_layers=None,
                       heads=None, scale=1.0):
    """transfer[i, j] over (src layer i, dst layer j).

    blank_prompts[u_index] : prompts with the f-slot replaced by a neutral token,
    so that whatever makes the operator appear has to come from the injected FV.

    Returns dict with `transfer` (n_src, n_dst), `null` (same shape), and
    `ratio` = (1 - null) normalised gain. ratio near 0 means the FV carried
    nothing beyond what any FV would carry."""
    readout_layer = readout_layer or probe.n_layers
    src_layers = src_layers or list(range(1, probe.n_layers))
    dst_layers = dst_layers or list(range(1, probe.n_layers))

    M_ref = {}
    for iu, u in enumerate(codes_u):
        ids, x_pos, rp = prompts[iu][0]
        M_ref[iu] = probe.M(ids, x_pos, rp, readout_layer)[0]

    n_s, n_d = len(src_layers), len(dst_layers)
    T = np.zeros((n_s, n_d))
    N = np.zeros((n_s, n_d))
    for si, ls in enumerate(src_layers):
        fvs = {iu: extract_fv(probe, prompts[iu], ls, heads)
               for iu in range(len(codes_u))}
        for di, ld in enumerate(dst_layers):
            hit, null = [], []
            for iu in range(len(codes_u)):
                ids, x_pos, rp = blank_prompts[iu][0]
                Mp = probe.M(ids, x_pos, rp, readout_layer,
                             patch=(ld, rp, scale * fvs[iu]))[0]
                ref = M_ref[iu]
                hit.append(1 - np.linalg.norm(Mp - ref) / (np.linalg.norm(ref) + EPS))
                jw = (iu + 1) % len(codes_u)
                Mn = probe.M(ids, x_pos, rp, readout_layer,
                             patch=(ld, rp, scale * fvs[jw]))[0]
                null.append(1 - np.linalg.norm(Mn - ref) / (np.linalg.norm(ref) + EPS))
            T[si, di] = np.mean(hit)
            N[si, di] = np.mean(null)
    return dict(src_layers=src_layers, dst_layers=dst_layers,
                transfer=T, null=N, gain=T - N,
                best=dict(zip(("src", "dst", "gain"),
                              (src_layers[int(np.argmax(T - N) // n_d)],
                               dst_layers[int(np.argmax(T - N) % n_d)],
                               float((T - N).max())))))
