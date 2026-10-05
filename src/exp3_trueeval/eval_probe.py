"""
eval_probe.py -- reference model, expansion points, and the (l_in, l_out) probe.

Two changes over the first version, both forced by measurements:

(1) EXPANSION POINT.  M is a linearisation, so it depends on where you
    linearise, and that dependence was the only gauge/stability check that
    failed: re-linearising at random points moved the effective rank from 2.96
    to 9.47 (pr_cv = 0.30) on an untrained model. The default is now option (b):
    shrink the one-hot argument coordinate toward the CENTROID of the value-token
    simplex, and average the Jacobian over a few jittered draws around it, the
    way LRE takes a mean Jacobian over n = 8 examples (their sweep plateaus at
    n = 5).

    The shrinkage parameter alpha matters and cannot be set to 0. With
    c0 independent of v the x-slot embedding is identical for every v, Sep is
    vacuously 0, and ANY model passes the factorisation test. ExpansionPoint
    refuses that configuration unless allow_vacuous is set explicitly.

(2) TWO-DIMENSIONAL LAYER SCAN.  The probe used to inject at the embedding and
    only sweep the readout layer. LRE searches the INJECTION layer per relation
    and finds optima spread from layer 1 to 11, so fixing it at 0 was a hole.
    M is now taken between an injection layer l_in and a readout layer l_out.

    Injection above layer 0 needs a frame in residual coordinates. build_in_basis
    supplies one: the hidden state at the x-slot when that slot carries each
    value token, averaged over prompts so the frame is FIXED and does not depend
    on u. Without that averaging, M(u) would be expressed in a u-dependent basis
    and r_op / rho would be meaningless.

    The readout is the logit-lens readout at l_out -- the decoder head restricted
    to the value tokens -- which is defined at every layer and shares one frame
    across the whole scan.
"""

import numpy as np

EPS = 1e-12


# ------------------------------------------------------------------ reference model

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _rmsnorm(x):
    """No per-channel gain -- it is folded into the following weight matrix.
    That folding is what makes the residual rotation gauge exact rather than
    approximate (the same preprocessing SliceGPT and QuaRot rely on)."""
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)


class TinyLM:
    """Row-vector convention: h is (T, d); every weight acts on the right."""

    def __init__(self, V=64, d=32, L=4, H=4, f=64, T=24, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda *sh: rng.normal(size=sh) / np.sqrt(sh[-2])
        self.V, self.d, self.L, self.H, self.T = V, d, L, H, T
        self.dh = d // H
        self.E, self.P = s(V, d), s(T, d) * 0.1
        self.Wq, self.Wk, self.Wv = s(L, H, d, self.dh), s(L, H, d, self.dh), s(L, H, d, self.dh)
        self.Wo, self.W1, self.W2 = s(L, d, d), s(L, d, f), s(L, f, d)
        self.WU = s(d, V)

    # -- gauge transforms ------------------------------------------------

    def rotated(self, Q):
        m = TinyLM.__new__(TinyLM)
        m.__dict__.update(self.__dict__)
        m.E, m.P = self.E @ Q, self.P @ Q
        m.Wq = np.einsum("ij,lhjk->lhik", Q.T, self.Wq)
        m.Wk = np.einsum("ij,lhjk->lhik", Q.T, self.Wk)
        m.Wv = np.einsum("ij,lhjk->lhik", Q.T, self.Wv)
        m.Wo = np.einsum("lij,jk->lik", self.Wo, Q)
        m.W1 = np.einsum("ij,ljk->lik", Q.T, self.W1)
        m.W2 = np.einsum("lij,jk->lik", self.W2, Q)
        m.WU = Q.T @ self.WU
        return m

    def head_permuted(self, perm):
        m = TinyLM.__new__(TinyLM)
        m.__dict__.update(self.__dict__)
        m.Wq, m.Wk, m.Wv = self.Wq[:, perm], self.Wk[:, perm], self.Wv[:, perm]
        idx = np.concatenate([np.arange(h * self.dh, (h + 1) * self.dh) for h in perm])
        m.Wo = self.Wo[:, idx, :]
        return m

    # -- forward ---------------------------------------------------------

    def run(self, ids, inject=None, patch=None, upto=None):
        """inject = (layer, pos, vector): REPLACE the residual stream there.
        patch  = (layer, pos, vector): ADD to the residual stream there.
        Returns hs, a list of L+1 states (hs[l] is the input to block l)."""
        T = len(ids)
        h = self.E[ids] + self.P[:T]
        mask = np.triu(np.full((T, T), -1e9), 1)
        L = self.L if upto is None else upto
        hs = []
        for l in range(L + 1):
            if inject is not None and inject[0] == l:
                h = h.copy()
                h[inject[1]] = inject[2]
            if patch is not None and patch[0] == l:
                h = h.copy()
                h[patch[1]] = h[patch[1]] + patch[2]
            hs.append(h)
            if l == L:
                break
            hn = _rmsnorm(h)
            outs = []
            for hd in range(self.H):
                q, k, v = hn @ self.Wq[l, hd], hn @ self.Wk[l, hd], hn @ self.Wv[l, hd]
                a = _softmax(q @ k.T / np.sqrt(self.dh) + mask)
                outs.append(a @ v)
            h = h + np.concatenate(outs, axis=-1) @ self.Wo[l]
            h = h + _gelu(_rmsnorm(h) @ self.W1[l]) @ self.W2[l]
        return hs

    def attentions(self, ids, inject=None):
        hs = self.run(ids, inject=inject)
        T = len(ids)
        mask = np.triu(np.full((T, T), -1e9), 1)
        A = np.zeros((self.L, self.H, T, T))
        for l in range(self.L):
            hn = _rmsnorm(hs[l])
            for hd in range(self.H):
                q, k = hn @ self.Wq[l, hd], hn @ self.Wk[l, hd]
                A[l, hd] = _softmax(q @ k.T / np.sqrt(self.dh) + mask)
        return A

    def head_out(self, ids, layer, pos, inject=None):
        hs = self.run(ids, inject=inject)
        T = len(ids)
        mask = np.triu(np.full((T, T), -1e9), 1)
        hn = _rmsnorm(hs[layer])
        outs = []
        for hd in range(self.H):
            q, k, v = (hn @ self.Wq[layer, hd], hn @ self.Wk[layer, hd],
                       hn @ self.Wv[layer, hd])
            a = _softmax(q @ k.T / np.sqrt(self.dh) + mask)
            z = np.zeros((T, self.d))
            z[:, hd * self.dh:(hd + 1) * self.dh] = a @ v
            outs.append((z @ self.Wo[layer])[pos])
        return np.stack(outs)


# ------------------------------------------------------------------ expansion point

class ExpansionPoint:
    """Where to linearise, in value-token coordinates c (length k).

    mode:
      "shrink"   c0(v) = (1-alpha) * centroid + alpha * onehot(v)    [default]
      "onehot"   c0(v) = onehot(v)                     (alpha = 1)
      "centroid" c0(v) = centroid                      (alpha = 0) -- VACUOUS

    The centroid is the uniform mixture of the value tokens, i.e. the barycentre
    of the simplex the argument slot ranges over. Shrinking toward it keeps the
    linearisation away from the corners, where the model is most extremal and the
    Jacobian least stable, while alpha > 0 preserves the v-dependence that Sep
    needs.

    n_jitter / sigma average the Jacobian over draws c0 + sigma * N(0, I), which
    is the mean-Jacobian estimator LRE uses. Set n_jitter = 1 to disable.
    """

    def __init__(self, k, mode="shrink", alpha=0.5, n_jitter=8, sigma=0.05,
                 seed=0, allow_vacuous=False):
        if mode == "onehot":
            alpha = 1.0
        if mode == "centroid":
            alpha = 0.0
        if alpha <= 0 and not allow_vacuous:
            raise ValueError(
                "alpha = 0 makes c0 independent of v: the x-slot embedding is "
                "then identical for every argument, Sep is vacuously 0 and any "
                "model passes. Use mode='shrink' with alpha > 0, or pass "
                "allow_vacuous=True if you are deliberately measuring that.")
        self.k, self.mode, self.alpha = k, mode, float(alpha)
        self.n_jitter, self.sigma = int(n_jitter), float(sigma)
        self.rng = np.random.default_rng(seed)
        self.centroid = np.full(k, 1.0 / k)
        self.offset = np.zeros(k)

    def displace(self, rng, scale):
        """Move the whole operating point. Used by G4 to ask how much the
        invariants depend on WHERE we linearise -- which is the question, and is
        distinct from how reproducible one linearisation is. Jitter averaging is
        supposed to shrink the answer; with n_jitter = 1 it cannot."""
        self.offset = scale * rng.normal(size=self.k)

    def base(self, v):
        c = self.centroid.copy() * (1.0 - self.alpha)
        c[v] += self.alpha
        return c + self.offset

    def draws(self, v):
        c = self.base(v)
        if self.n_jitter <= 1 or self.sigma <= 0:
            return [c]
        return [c] + [c + self.sigma * self.rng.normal(size=self.k)
                      for _ in range(self.n_jitter - 1)]

    def describe(self):
        return dict(mode=self.mode, alpha=self.alpha,
                    n_jitter=self.n_jitter, sigma=self.sigma)


# ------------------------------------------------------------------ frames

def build_in_basis(model, prompts, layer, x_pos, value_ids):
    """Fixed frame in residual coordinates at `layer`, one direction per value
    token: the hidden state at the x-slot when that slot carries the token,
    averaged over prompts.

    Averaging over prompts is what makes the frame independent of u. A per-prompt
    frame would express each M(u) in its own coordinates and silently destroy
    r_op and rho.
    """
    k = len(value_ids)
    B = np.zeros((k, model.d))
    for i, tok in enumerate(value_ids):
        acc = np.zeros(model.d)
        for ids in prompts:
            ids = np.asarray(ids).copy()
            ids[x_pos] = tok
            acc += model.run(ids, upto=layer)[layer][x_pos]
        B[i] = acc / len(prompts)
    return B


def readout_frame(model, value_ids):
    """Logit-lens readout: decoder head restricted to the value tokens. Defined
    at every layer, so one frame serves the whole (l_in, l_out) scan."""
    return model.WU[:, np.asarray(value_ids)].T          # (k, d)


# ------------------------------------------------------------------ the probe

class TinyProbe:
    def __init__(self, model, B_in_of_layer, R_out, eps=1e-4):
        """B_in_of_layer: dict {layer -> (k, d)} frames from build_in_basis."""
        self.m = model
        self.B = B_in_of_layer
        self.R = R_out
        self.eps = eps
        self.n_layers = model.L
        self.k = R_out.shape[0]

    def _readout(self, hs, l_out, read_pos):
        return self.R @ _rmsnorm(hs[l_out])[read_pos]

    def M(self, ids, x_pos, read_pos, l_in, l_out, ep=None, v=0, patch=None,
          return_spread=False):
        """M = d(value-token logits at l_out) / d c, injected at l_in.

        ep: an ExpansionPoint. The Jacobian is averaged over its draws, and the
        spread across draws is reported so instability is visible rather than
        silently averaged away.
        """
        if l_out < l_in:
            raise ValueError("l_out must be >= l_in")
        B = self.B[l_in]
        k = B.shape[0]
        cs = ep.draws(v) if ep is not None else [np.eye(k)[v]]

        Ms, bs = [], []
        for c0 in cs:
            def f(c):
                hs = self.m.run(ids, inject=(l_in, x_pos, c @ B), patch=patch,
                                upto=l_out)
                return self._readout(hs, l_out, read_pos)
            cols = []
            for i in range(k):
                cp, cm = c0.copy(), c0.copy()
                cp[i] += self.eps
                cm[i] -= self.eps
                cols.append((f(cp) - f(cm)) / (2 * self.eps))
            Mi = np.stack(cols, axis=1)
            Ms.append(Mi)
            bs.append(f(c0) - Mi @ c0)

        M = np.mean(Ms, axis=0)
        b = np.mean(bs, axis=0)
        if return_spread:
            spread = float(np.std([np.linalg.norm(x - M) for x in Ms])
                           / (np.linalg.norm(M) + EPS)) if len(Ms) > 1 else 0.0
            return M, b, spread
        return M, b

    def head_out(self, ids, layer, pos):
        return self.m.head_out(ids, layer, pos % len(ids))


# ------------------------------------------------------------------ gauge suite

def _probe_for(model, prompts, x_pos, value_ids, layers):
    B = {l: build_in_basis(model, prompts, l, x_pos, value_ids) for l in layers}
    return TinyProbe(model, B, readout_frame(model, value_ids))


def check_G1(pr, ids, x_pos, read_pos, l_in, l_out, ep, rng):
    """Probe-side covariance: M' =? T M S^{-1}. An identity, so this is an
    instrument check (transposes, signs, expansion points), not a measurement."""
    k = pr.k
    S = rng.normal(size=(k, k)) / np.sqrt(k) + np.eye(k)
    T = rng.normal(size=(k, k)) / np.sqrt(k) + np.eye(k)
    Si = np.linalg.inv(S)
    M, _ = pr.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0)
    pr2 = TinyProbe(pr.m, {l_in: Si.T @ pr.B[l_in]}, T @ pr.R, pr.eps)
    ep2 = ExpansionPoint(k, mode=ep.mode, alpha=ep.alpha, n_jitter=1)
    # express the same operating point in the new coordinates
    c_new = S @ ep.base(0)
    Mp = _jac_at(pr2, ids, x_pos, read_pos, l_in, l_out, c_new)
    pred = T @ M @ Si
    rel = np.linalg.norm(Mp - pred) / (np.linalg.norm(pred) + EPS)
    return dict(check="G1_probe_covariance", rel_err=float(rel),
                passed=bool(rel < 1e-3))


def _jac_at(pr, ids, x_pos, read_pos, l_in, l_out, c0):
    B, k = pr.B[l_in], len(c0)

    def f(c):
        hs = pr.m.run(ids, inject=(l_in, x_pos, c @ B), upto=l_out)
        return pr._readout(hs, l_out, read_pos)
    cols = []
    for i in range(k):
        cp, cm = c0.copy(), c0.copy()
        cp[i] += pr.eps
        cm[i] -= pr.eps
        cols.append((f(cp) - f(cm)) / (2 * pr.eps))
    return np.stack(cols, axis=1)


def check_G2(model, prompts, ids, x_pos, read_pos, l_in, l_out, value_ids, ep, rng):
    """Model-side residual rotation. M and A must be IDENTICAL, not similar."""
    Q, _ = np.linalg.qr(rng.normal(size=(model.d, model.d)))
    rot = model.rotated(Q)
    p0 = _probe_for(model, prompts, x_pos, value_ids, [l_in])
    p1 = _probe_for(rot, prompts, x_pos, value_ids, [l_in])
    M0, _ = p0.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0)
    M1, _ = p1.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0)
    A0, A1 = model.attentions(ids), rot.attentions(ids)
    em = np.linalg.norm(M1 - M0) / (np.linalg.norm(M0) + EPS)
    ea = float(np.abs(A1 - A0).max())
    return dict(check="G2_residual_rotation", rel_err_M=float(em),
                max_abs_err_A=ea, passed=bool(em < 1e-6 and ea < 1e-8))


def check_G3(model, prompts, ids, x_pos, read_pos, l_in, l_out, value_ids, ep, rng):
    perm = rng.permutation(model.H)
    pm = model.head_permuted(perm)
    p0 = _probe_for(model, prompts, x_pos, value_ids, [l_in])
    p1 = _probe_for(pm, prompts, x_pos, value_ids, [l_in])
    M0, _ = p0.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0)
    M1, _ = p1.M(ids, x_pos, read_pos, l_in, l_out, ep, v=0)
    em = np.linalg.norm(M1 - M0) / (np.linalg.norm(M0) + EPS)
    return dict(check="G3_head_permutation", rel_err_M=float(em),
                passed=bool(em < 1e-6))


def check_G4(pr, prompt_rows, x_pos, read_pos, l_in, l_out, ep, n_rep=6,
             rep_sigma=0.3, seed=0):
    """Operating-point stability. NOT a gauge test: this is the measurement the
    expansion-point design exists to fix. Each repetition moves the whole
    operating point by rep_sigma and re-estimates; the coefficient of variation
    across repetitions says how much of r_op is a property of the model and how
    much is a property of where we happened to linearise."""
    from eval_common import effective_rank, intrinsic_dim_twonn
    rng = np.random.default_rng(seed)
    dims, prs = [], []
    for _ in range(n_rep):
        ep.displace(rng, rep_sigma)
        Mu = []
        for row in prompt_rows:
            Ms = [pr.M(ids, x_pos, read_pos, l_in, l_out, ep, v=iv)[0]
                  for iv, ids in enumerate(row)]
            Mu.append(np.mean(Ms, axis=0))
        Mu = np.asarray(Mu)
        prs.append(effective_rank(Mu)["pr"])
        dims.append(intrinsic_dim_twonn(Mu)["id_twonn"])
    ep.offset = np.zeros(ep.k)
    prs, dims = np.asarray(prs), np.asarray(dims)
    cv_pr = float(prs.std() / (prs.mean() + EPS))
    cv_id = float(np.nanstd(dims) / (np.nanmean(dims) + EPS))
    return dict(check="G4_operating_point", pr_mean=float(prs.mean()),
                pr_cv=cv_pr, id_mean=float(np.nanmean(dims)), id_cv=cv_id,
                rep_sigma=rep_sigma, expansion=ep.describe(),
                passed=bool(cv_pr < 0.15))
