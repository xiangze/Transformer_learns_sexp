"""calibrate.py -- noise calibration for the eval probes.

Given a model whose Jacobian error is  eps = |M_hat - M_GT| / |M_GT|,
what values of Sep / sep_spec / rho / id_op does a GENUINELY functorial
model produce?  Without this, a nonzero rho cannot be read: some of it is
always just fitting error propagating through the composition.

Two noise channels are simulated separately, because they mean different
things:

  eps_u : M_hat(u) = A_u + eps_u E_u          systematic per-program error.
          Sep is unaffected (no v-dependence); rho IS affected.
  eps_v : M_hat(u,v) = ... + eps_v E_{u,v}    v-dependence leaking in.
          Sep drops; this is the quantity Sep was designed to detect.

Analytic expectation for rho under pure eps_u noise, to first order:
    M(uu') - M(u)M(u') = eps(E_uu' - E_u A_u' - A_u E_u') + O(eps^2)
    -> rho_lin ~ sqrt(3) * eps_u     (A orthogonal, E independent)
This script checks that against simulation.
"""
import numpy as np
from probe_mlp import sep_eta2, sep_spec, rho_linear, rho_spectral, twonn, whiteners, whiten
from sexp_cont import N_DIM, program_matrix, sample_composable_pair, sample_program, compose

def calib(eps_u, eps_v, n_u=40, n_v=10, n_pairs=200, seed=0):
    rng = np.random.default_rng(seed)
    progs = [sample_program(rng) for _ in range(n_u)]
    A = np.stack([program_matrix(u) for u in progs])
    nrm = np.sqrt(N_DIM)                       # ||A||_F for orthogonal A
    Eu = rng.normal(size=(n_u, N_DIM, N_DIM)); Eu /= np.linalg.norm(Eu, axis=(1,2), keepdims=True)
    Euv = rng.normal(size=(n_u, n_v, N_DIM, N_DIM))
    Euv /= np.linalg.norm(Euv, axis=(2,3), keepdims=True)
    M = A[:, None] + eps_u * nrm * Eu[:, None] + eps_v * nrm * Euv

    Wy, Wx = whiteners(M.reshape(-1, N_DIM, N_DIM))
    Mw = np.stack([[whiten(M[i,k], Wy, Wx) for k in range(n_v)] for i in range(n_u)])

    # rho: independent composable pairs, same noise law
    rl, rs = [], []
    cache = {}
    def Mhat(u):
        if u not in cache:
            E = rng.normal(size=(N_DIM, N_DIM)); E /= np.linalg.norm(E)
            cache[u] = program_matrix(u) + eps_u * nrm * E
        return cache[u]
    for _ in range(n_pairs):
        a, b = sample_composable_pair(rng)
        ab = compose(a, b)
        rl.append(rho_linear(Mhat(ab), Mhat(a), Mhat(b)))
        rs.append(rho_spectral(Mhat(ab), Mhat(a), Mhat(b)))
    return dict(Sep=sep_eta2(Mw), sep_spec=sep_spec(Mw, pairs=24, seed=seed),
                id_op=twonn(M[:,0].reshape(n_u,-1)),
                rho_lin=float(np.mean(rl)), rho_spec=float(np.nanmean(rs)),
                rho_pred=np.sqrt(3)*eps_u)

print("Functorial-with-noise null model (what a TRUE eval model should show)")
print(f"{'eps_u':>7s} {'eps_v':>7s} | {'Sep':>7s} {'sep_spec':>9s} {'id_op':>6s} "
      f"{'rho_lin':>8s} {'rho_pred':>9s} {'rho_spec':>9s}")
print("-"*76)
for eps in [0.0, 0.003, 0.012, 0.03, 0.1, 0.3]:
    r = calib(eps, eps)
    print(f"{eps:7.3f} {eps:7.3f} | {r['Sep']:7.4f} {r['sep_spec']:9.4f} {r['id_op']:6.2f} "
          f"{r['rho_lin']:8.4f} {r['rho_pred']:9.4f} {r['rho_spec']:9.4f}")

print("\nSplit channels: eps_u only (no v-leakage) vs eps_v only")
print(f"{'case':>18s} | {'Sep':>7s} {'sep_spec':>9s} {'rho_lin':>8s}")
print("-"*50)
for name, (eu, ev) in [("eps_u=0.012 only", (0.012, 0.0)),
                       ("eps_v=0.012 only", (0.0, 0.012)),
                       ("eps_u=0.10  only", (0.10, 0.0)),
                       ("eps_v=0.10  only", (0.0, 0.10))]:
    r = calib(eu, ev)
    print(f"{name:>18s} | {r['Sep']:7.4f} {r['sep_spec']:9.4f} {r['rho_lin']:8.4f}")
