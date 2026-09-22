"""sexp_cont.py -- continuous-argument S-expression probe language.

Design constraints (see accompanying notes):

  * The PROGRAM u is a discrete token string (S-expression).
  * The ARGUMENT v is a continuous vector in R^N_DIM, injected through a
    separate value channel at <NUM> placeholder positions.
  * Every program denotes a LINEAR map on R^N_DIM.  This is deliberate:
    under the SMCC / linear-lambda reading, true eval predicts that
    M(u,v) = dF/dv is independent of v *exactly*.  Any residual
    v-dependence is leakage from MLP nonlinearity, and is measurable.
  * Primitives are orthogonal (rotations / reflections), so composition
    stays well conditioned and T = M(u')^{-1} M(u) has spectrum on the
    unit circle.  The operator family {M(u)} therefore lies on O(N_DIM),
    an intrinsic-dimension ground truth of dim SO(3) = 3 for N_DIM = 3.

Program grammar:

    u    ::= prim | ( comp u ... u )        # leftmost applied LAST
    prim ::= ( rot PLANE ANGLE ) | ( refl AXIS )

Serialized example:

    ( comp ( rot p01 a3 ) ( refl x2 ) ) | <NUM> <NUM> <NUM> => <OUT> <OUT> <OUT>
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# language constants
# --------------------------------------------------------------------------
N_DIM = 3
N_ANGLE = 12
PLANES: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]
MAX_DEPTH = 3

Prim = Tuple  # ('rot', i, j, a) | ('refl', i)
Program = Tuple[Prim, ...]


def prim_matrix(p: Prim) -> np.ndarray:
    if p[0] == "rot":
        _, i, j, a = p
        th = 2.0 * np.pi * a / N_ANGLE
        M = np.eye(N_DIM)
        c, s = np.cos(th), np.sin(th)
        M[i, i] = c
        M[i, j] = -s
        M[j, i] = s
        M[j, j] = c
        return M
    if p[0] == "refl":
        M = np.eye(N_DIM)
        M[p[1], p[1]] = -1.0
        return M
    raise ValueError(p)


def program_matrix(u: Program) -> np.ndarray:
    """Matrix of u.  Leftmost primitive is applied LAST (function composition)."""
    M = np.eye(N_DIM)
    for p in u:
        M = M @ prim_matrix(p)
    return M


def all_prims() -> List[Prim]:
    out: List[Prim] = []
    for (i, j) in PLANES:
        for a in range(1, N_ANGLE):  # drop a=0 (identity) to avoid degenerate prims
            out.append(("rot", i, j, a))
    for i in range(N_DIM):
        out.append(("refl", i))
    return out


PRIMS = all_prims()

# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
SPECIALS = ["<pad>", "(", ")", "comp", "|", "=>", "<NUM>", "<OUT>"]
PRIM_TOKS = ["rot", "refl"]
PLANE_TOKS = [f"p{i}{j}" for (i, j) in PLANES]
ANGLE_TOKS = [f"a{a}" for a in range(N_ANGLE)]
AXIS_TOKS = [f"x{i}" for i in range(N_DIM)]
VOCAB = SPECIALS + PRIM_TOKS + PLANE_TOKS + ANGLE_TOKS + AXIS_TOKS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = STOI["<pad>"]
NUM_ID = STOI["<NUM>"]
OUT_ID = STOI["<OUT>"]
VOCAB_SIZE = len(VOCAB)


def prim_tokens(p: Prim) -> List[str]:
    if p[0] == "rot":
        _, i, j, a = p
        return ["(", "rot", f"p{i}{j}", f"a{a}", ")"]
    return ["(", "refl", f"x{p[1]}", ")"]


def program_tokens(u: Program) -> List[str]:
    if len(u) == 1:
        return prim_tokens(u[0])
    toks = ["(", "comp"]
    for p in u:
        toks += prim_tokens(p)
    toks += [")"]
    return toks


# longest sequence: MAX_DEPTH prims inside a comp
_MAX_PROG = 2 + 5 * MAX_DEPTH + 1
SEQ_LEN = _MAX_PROG + 1 + N_DIM + 1 + N_DIM


def encode(u: Program) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, num_mask, out_positions) padded to SEQ_LEN.

    Padding is placed *before* the program so that the <NUM> / <OUT> block
    always sits at fixed positions at the tail of the sequence.  This keeps
    readout indices constant across programs of different length.
    """
    toks = program_tokens(u) + ["|"] + ["<NUM>"] * N_DIM + ["=>"] + ["<OUT>"] * N_DIM
    pad = SEQ_LEN - len(toks)
    assert pad >= 0, (len(toks), SEQ_LEN)
    toks = ["<pad>"] * pad + toks
    ids = np.array([STOI[t] for t in toks], dtype=np.int64)
    num_mask = ids == NUM_ID
    out_pos = np.where(ids == OUT_ID)[0]
    return ids, num_mask, out_pos


# --------------------------------------------------------------------------
# program sampling
# --------------------------------------------------------------------------
def sample_program(rng: np.random.Generator, depth: int | None = None) -> Program:
    d = depth if depth is not None else int(rng.integers(1, MAX_DEPTH + 1))
    idx = rng.integers(0, len(PRIMS), size=d)
    return tuple(PRIMS[i] for i in idx)


def sample_composable_pair(rng: np.random.Generator) -> Tuple[Program, Program]:
    """(u, u') with len(u) + len(u') <= MAX_DEPTH, so u o u' is in-distribution."""
    d1 = int(rng.integers(1, MAX_DEPTH))
    d2 = int(rng.integers(1, MAX_DEPTH - d1 + 1))
    return sample_program(rng, d1), sample_program(rng, d2)


def compose(u: Program, u2: Program) -> Program:
    """u o u' : apply u' first.  Leftmost applied last, so concatenation works."""
    return tuple(u) + tuple(u2)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
@dataclass
class Batch:
    ids: np.ndarray       # [B, T] int64
    vals: np.ndarray      # [B, T] float32, nonzero only at <NUM>
    y: np.ndarray         # [B, N_DIM] float32
    programs: List[Program]


GRID = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float64)


def make_batch(
    rng: np.random.Generator,
    bsz: int,
    mode: str = "eval",
    fixed_u: Program | None = None,
) -> Batch:
    """mode:
        'eval'        -- y = M(u) v, v continuous               (true eval)
        'grid'        -- y = M(u) v, v from a finite grid        (lookup-able control)
        'ignore_code' -- y = M(u0) v regardless of u             (code-blind control)
    """
    ids = np.zeros((bsz, SEQ_LEN), dtype=np.int64)
    vals = np.zeros((bsz, SEQ_LEN), dtype=np.float32)
    ys = np.zeros((bsz, N_DIM), dtype=np.float32)
    progs: List[Program] = []
    u0 = fixed_u if fixed_u is not None else (PRIMS[0],)

    for b in range(bsz):
        u = sample_program(rng)
        i, nm, _ = encode(u)
        if mode == "grid":
            v = rng.choice(GRID, size=N_DIM)
        else:
            v = rng.normal(0.0, 1.0, size=N_DIM)
        M = program_matrix(u0 if mode == "ignore_code" else u)
        ids[b] = i
        vals[b, nm] = v.astype(np.float32)
        ys[b] = (M @ v).astype(np.float32)
        progs.append(u)
    return Batch(ids, vals, ys, progs)


def encode_with_v(u: Program, v: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    ids, nm, _ = encode(u)
    vals = np.zeros(SEQ_LEN, dtype=np.float32)
    vals[nm] = np.asarray(v, dtype=np.float32)
    return ids, vals


def num_positions() -> np.ndarray:
    ids, nm, _ = encode((PRIMS[0],))
    return np.where(nm)[0]


def out_positions() -> np.ndarray:
    ids, _, op = encode((PRIMS[0],))
    return op


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    u = sample_program(rng)
    print("program :", u)
    print("tokens  :", " ".join(program_tokens(u)))
    print("matrix  :\n", np.round(program_matrix(u), 3))
    print("SEQ_LEN :", SEQ_LEN, " vocab:", VOCAB_SIZE)
    print("num pos :", num_positions(), " out pos:", out_positions())
    b = make_batch(rng, 4)
    print("batch ids", b.ids.shape, "y", b.y.shape)
    # sanity: composition is matrix product
    a, c = sample_composable_pair(rng)
    lhs = program_matrix(compose(a, c))
    rhs = program_matrix(a) @ program_matrix(c)
    print("compose ok:", np.allclose(lhs, rhs))
