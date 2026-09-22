"""
eval_gauge.py -- what quantity actually witnesses "M is a (1,1) tensor", and how
to check it on a real model.

There are two different tests and conflating them is the mistake to avoid.

  G1  PROBE-SIDE COVARIANCE.  The bases in which M is expressed are OUR choice:
      the x-slot is written e_x = sum_i c_i E[t_i] and the readout is the logit
      vector over the same tokens. Changing those bases, c' = S c and r' = T r,
      must give exactly
              M' = T M S^{-1}.
      This is an identity, so G1 is an INSTRUMENT check, not a measurement --
      it catches sign errors, transposes, wrong expansion points, and
      finite-difference bias. It cannot fail for an interesting reason.
      What it does establish is that r_op and rho, being functions of
      vec(M) up to the invertible map (S^{-T} kron T), are unchanged.

  G2  MODEL-SIDE GAUGE.  Rotate the residual stream, h -> h Q with Q orthogonal,
      and push Q through every weight. For an RMSNorm model with the per-channel
      gain folded into the following matrix, this leaves the input-output map
      EXACTLY invariant (this is the same observation SliceGPT and QuaRot use).
      Here M and A must come out bit-for-bit identical -- not similar, identical,
      because the input and output spaces were never touched. G2 CAN fail: it
      fails for LayerNorm models under general Q (mean subtraction fixes the
      all-ones direction, so only Q with Q1 = 1 are exact), and it fails if any
      statistic secretly reads a residual-stream coordinate.

  G3  HEAD PERMUTATION.  Invariant by construction here, since A is only ever
      used as a stacked set over (layer, head).

  G4  OPERATING POINT.  Not a gauge transformation at all, and the one that
      matters most: M is a linearisation, so it depends on where you linearise.
      The invariants must be stable when the expansion point moves. Nothing
      guarantees this, and it is a genuine measurement.

TinyLM below is a numpy reference transformer used to exercise all of this with
no GPU and no checkpoint -- the gauge analogue of the stage-0 oracles.
"""

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- reference model

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _rmsnorm(x):
    """No per-channel gain: it is folded into the following weight matrix.
    That folding is what makes the rotation gauge exact rather than approximate."""
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)


class TinyLM:
    """Row-vector convention throughout: h is (T, d), all weights act on the right.

    attn:  q = rmsnorm(h) Wq[l,hd] ...   out = concat_heads Wo[l]
    mlp:   rmsnorm(h) W1[l] -> gelu -> W2[l]
    head:  logits = rmsnorm(h) WU
    """

    def __init__(self, V=64, d=32, L=3, H=4, f=64, T=16, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda *sh: rng.normal(size=sh) / np.sqrt(sh[-2])
        self.V, self.d, self.L, self.H, self.T = V, d, L, H, T
        self.dh = d // H
        self.E = s(V, d)
        self.P = s(T, d) * 0.1
        self.Wq = s(L, H, d, self.dh)
        self.Wk = s(L, H, d, self.dh)
        self.Wv = s(L, H, d, self.dh)
        self.Wo = s(L, d, d)
        self.W1 = s(L, d, f)
        self.W2 = s(L, f, d)
        self.WU = s(d, V)

    # -- gauge transforms -------------------------------------------------

    def rotated(self, Q):
        """G2. h -> h Q. Exactly function-preserving for this architecture."""
        m = TinyLM.__new__(TinyLM)
        m.__dict__.update({k: v for k, v in self.__dict__.items()})
        m.E = self.E @ Q
        m.P = self.P @ Q
        m.Wq = np.einsum("ij,lhjk->lhik", Q.T, self.Wq)
        m.Wk = np.einsum("ij,lhjk->lhik", Q.T, self.Wk)
        m.Wv = np.einsum("ij,lhjk->lhik", Q.T, self.Wv)
        m.Wo = np.einsum("lij,jk->lik", self.Wo, Q)
        m.W1 = np.einsum("ij,ljk->lik", Q.T, self.W1)
        m.W2 = np.einsum("lij,jk->lik", self.W2, Q)
        m.WU = Q.T @ self.WU
        return m

    def head_permuted(self, perm):
        """G3."""
        m = TinyLM.__new__(TinyLM)
        m.__dict__.update({k: v for k, v in self.__dict__.items()})
        m.Wq, m.Wk, m.Wv = self.Wq[:, perm], self.Wk[:, perm], self.Wv[:, perm]
        # Wo consumes concat(heads); permute its row blocks to match
        idx = np.concatenate([np.arange(h * self.dh, (h + 1) * self.dh)
                              for h in perm])
        m.Wo = self.Wo[:, idx, :]
        return m

    # -- forward ----------------------------------------------------------

    def forward(self, embeds, need_attn=False, upto=None):
        """embeds (T,d) -> (hidden_states list of L+1 (T,d), attn (L,H,T,T)|None)"""
        T = embeds.shape[0]
        h = embeds + self.P[:T]
        hs, attns = [h], []
        mask = np.triu(np.full((T, T), -1e9), 1)
        L = self.L if upto is None else upto
        for l in range(L):
            hn = _rmsnorm(h)
            outs = []
            for hd in range(self.H):
                q, k, v = hn @ self.Wq[l, hd], hn @ self.Wk[l, hd], hn @ self.Wv[l, hd]
                a = _softmax(q @ k.T / np.sqrt(self.dh) + mask)
                outs.append(a @ v)
                if need_attn:
                    attns.append(a)
            h = h + np.concatenate(outs, axis=-1) @ self.Wo[l]
            h = h + _gelu(_rmsnorm(h) @ self.W1[l]) @ self.W2[l]
            hs.append(h)
        A = (np.asarray(attns).reshape(L, self.H, T, T) if need_attn else None)
        return hs, A

    def logits(self, embeds, layer=-1):
        hs, _ = self.forward(embeds)
        return _rmsnorm(hs[layer]) @ self.WU


# ---------------------------------------------------------------- M in a chosen basis

def jac_M(model, ids, x_pos, B_in, R_out, read_pos=-1, layer=-1, c0=None,
          eps=1e-4):
    """M = d (R_out . h_read) / d c, central differences.

    B_in  (k_in, d)   directions injected at the x-slot:  e_x = c @ B_in
    R_out (k_out, d)  readout directions applied to the hidden state

    Changing B_in -> S^{-T} B_in and R_out -> T R_out is exactly the basis change
    c' = S c, r' = T r, so G1 predicts M' = T M S^{-1}.
    """
    k_in = B_in.shape[0]
    c0 = np.zeros(k_in) if c0 is None else np.asarray(c0, float)

    def fwd(c):
        e = model.E[ids].copy()
        e[x_pos] = c @ B_in
        hs, _ = model.forward(e)
        return R_out @ _rmsnorm(hs[layer])[read_pos]

    cols = []
    for i in range(k_in):
        cp, cm = c0.copy(), c0.copy()
        cp[i] += eps
        cm[i] -= eps
        cols.append((fwd(cp) - fwd(cm)) / (2 * eps))
    M = np.stack(cols, axis=1)
    return M, fwd(c0) - M @ c0


def attn_A(model, ids, x_pos, B_in, c0=None):
    k = B_in.shape[0]
    c0 = np.zeros(k) if c0 is None else c0
    e = model.E[ids].copy()
    e[x_pos] = c0 @ B_in
    _, A = model.forward(e, need_attn=True)
    return A


# ---------------------------------------------------------------- the four checks

def check_G1(model, ids, x_pos, B_in, R_out, rng, read_pos=-1):
    """Probe-side covariance: M' =? T M S^{-1}."""
    k_in, k_out = B_in.shape[0], R_out.shape[0]
    S = rng.normal(size=(k_in, k_in)) / np.sqrt(k_in) + np.eye(k_in)
    T = rng.normal(size=(k_out, k_out)) / np.sqrt(k_out) + np.eye(k_out)
    Si = np.linalg.inv(S)

    M, _ = jac_M(model, ids, x_pos, B_in, R_out, read_pos)
    # c' = S c  =>  e_x = c' (S^{-T} B_in);  r' = T r  =>  R_out' = T R_out
    Mp, _ = jac_M(model, ids, x_pos, Si.T @ B_in, T @ R_out, read_pos)
    pred = T @ M @ Si
    rel = np.linalg.norm(Mp - pred) / (np.linalg.norm(pred) + EPS)
    return dict(check="G1_probe_covariance", rel_err=float(rel),
                passed=bool(rel < 1e-3))


def check_G2(model, ids, x_pos, B_in, R_out, rng, read_pos=-1):
    """Model-side residual rotation: M and A must be IDENTICAL."""
    d = model.d
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    rot = model.rotated(Q)

    M, _ = jac_M(model, ids, x_pos, B_in, R_out, read_pos)
    # the probe bases live in residual coordinates, so they rotate with it
    Mr, _ = jac_M(rot, ids, x_pos, B_in @ Q, R_out @ Q, read_pos)
    A = attn_A(model, ids, x_pos, B_in)
    Ar = attn_A(rot, ids, x_pos, B_in @ Q)

    em = np.linalg.norm(Mr - M) / (np.linalg.norm(M) + EPS)
    ea = np.abs(Ar - A).max()
    return dict(check="G2_residual_rotation", rel_err_M=float(em),
                max_abs_err_A=float(ea),
                passed=bool(em < 1e-6 and ea < 1e-8))


def check_G3(model, ids, x_pos, B_in, R_out, rng, read_pos=-1):
    perm = rng.permutation(model.H)
    pm = model.head_permuted(perm)
    M, _ = jac_M(model, ids, x_pos, B_in, R_out, read_pos)
    Mp, _ = jac_M(pm, ids, x_pos, B_in, R_out, read_pos)
    A = np.sort(attn_A(model, ids, x_pos, B_in), axis=1)
    Ap = np.sort(attn_A(pm, ids, x_pos, B_in), axis=1)
    em = np.linalg.norm(Mp - M) / (np.linalg.norm(M) + EPS)
    return dict(check="G3_head_permutation", rel_err_M=float(em),
                max_abs_err_A_sorted=float(np.abs(Ap - A).max()),
                passed=bool(em < 1e-6))


def check_G4(model, ids_list, x_pos, B_in, R_out, rng, read_pos=-1, n_pts=6):
    """Operating-point stability. NOT a gauge test -- a measurement.

    Re-linearise at several random expansion points and report how much the
    invariants move. Large drift means M is not usefully a single operator and
    the whole linear reading is local only."""
    from eval_common import effective_rank, intrinsic_dim_twonn
    prs = []
    for _ in range(n_pts):
        c0 = rng.normal(size=B_in.shape[0]) * 0.3
        Ms = [jac_M(model, ids, x_pos, B_in, R_out, read_pos, c0=c0)[0]
              for ids in ids_list]
        prs.append(effective_rank(np.asarray(Ms))["pr"])
    prs = np.asarray(prs)
    cv = float(prs.std() / (prs.mean() + EPS))
    return dict(check="G4_operating_point", pr_mean=float(prs.mean()),
                pr_cv=cv, passed=bool(cv < 0.15), pr_all=prs.tolist())


def run_gauge_suite(model, ids_list, x_pos, B_in, R_out, seed=0, read_pos=-1):
    rng = np.random.default_rng(seed)
    ids = ids_list[0]
    out = [check_G1(model, ids, x_pos, B_in, R_out, rng, read_pos),
           check_G2(model, ids, x_pos, B_in, R_out, rng, read_pos),
           check_G3(model, ids, x_pos, B_in, R_out, rng, read_pos),
           check_G4(model, ids_list, x_pos, B_in, R_out, rng, read_pos)]
    return out
