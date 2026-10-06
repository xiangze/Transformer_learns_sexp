"""sep_fix.py -- replacement for sep_spec that does not sort eigenvalues.

Problem: spec_T sorts eigenvalues by angle.  For orthogonal T the spectrum is
{1, e^{i theta}, e^{-i theta}}; noise splits near-degenerate eigenvalues and
flips the sort order across v, so std-over-v has a floor (~0.12) that hides
small v-leakage.

Fix: use the coefficients of the characteristic polynomial of T.  They are
continuous in T, need no ordering, and are invariant under T -> S T S^-1,
which is exactly the GL freedom that survives after whitening.  For 3x3:
    c1 = tr T,  c2 = (tr(T)^2 - tr(T^2))/2,  c3 = det T
sep_inv = mean over (u,u') pairs of std_v(c) / (|mean_v c| + 1), averaged over
the three coefficients.  Zero iff T is exactly v-independent.
"""
import numpy as np
from probe_mlp import whiteners, whiten
from sexp_cont import N_DIM, program_matrix, sample_program


def charpoly_inv(T: np.ndarray) -> np.ndarray:
    """Characteristic-polynomial coefficients of T: conjugation-invariant, ordering-free."""
    n = T.shape[0]
    c = np.empty(n)
    Tk = np.eye(n)
    traces = []
    for k in range(1, n + 1):
        Tk = Tk @ T
        traces.append(np.trace(Tk))
    # Newton's identities -> elementary symmetric polynomials
    e = [1.0]
    for k in range(1, n + 1):
        s = 0.0
        for i in range(1, k + 1):
            s += (-1) ** (i - 1) * e[k - i] * traces[i - 1]
        e.append(s / k)
    return np.array(e[1:])


def sep_inv(M: np.ndarray, pairs: int = 64, seed: int = 0) -> float:
    """M: [n_u, n_v, m, n].  0 iff T(u,u') is v-independent."""
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
        C = np.stack(C)                                    # [n_v, 3]
        acc.append(np.mean(C.std(0) / (np.abs(C.mean(0)) + 1.0)))
    return float(np.mean(acc))


if __name__ == "__main__":
    from probe_mlp import sep_eta2, sep_spec
    rng = np.random.default_rng(0)
    n_u, n_v = 40, 10
    progs = [sample_program(rng) for _ in range(n_u)]
    A = np.stack([program_matrix(u) for u in progs]); nrm = np.sqrt(N_DIM)
    Eu = rng.normal(size=(n_u, N_DIM, N_DIM)); Eu /= np.linalg.norm(Eu, axis=(1,2), keepdims=True)
    Euv = rng.normal(size=(n_u, n_v, N_DIM, N_DIM))
    Euv /= np.linalg.norm(Euv, axis=(2,3), keepdims=True)

    print("v-leakage response (eps_u fixed at 0, only eps_v varies)")
    print(f"{'eps_v':>8s} | {'Sep':>8s} {'sep_spec(old)':>14s} {'sep_inv(new)':>13s}")
    print("-"*52)
    for ev in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        M = A[:, None] + ev * nrm * Euv
        Wy, Wx = whiteners(M.reshape(-1, N_DIM, N_DIM))
        Mw = np.stack([[whiten(M[i,k], Wy, Wx) for k in range(n_v)] for i in range(n_u)])
        print(f"{ev:8.3f} | {sep_eta2(Mw):8.5f} {sep_spec(Mw,pairs=24):14.4f} "
              f"{sep_inv(Mw,pairs=24):13.5f}")
