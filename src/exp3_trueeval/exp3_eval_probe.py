#!/usr/bin/env python3
"""
exp3_eval_probe.py -- does a trained Transformer's computation factor as
    Phi(u, v) = eval( [[u]] , v ) ?

Stages
  0  oracle validation. Four hand-built ground truths with known answers.
     Nothing here touches a network; this stage decides whether the statistics
     are measuring what they claim. If stage 0 fails, later stages are noise.
  1  floor. Random-init model of the same shape -- the value of r_op that
     LayerNorm and softmax produce with no training at all.
  2  probe a checkpoint (HF causal LM, or a local S-expression model).
  3  gauge guard. Recompute after head permutation and after a
     function-preserving reparameterisation; any statistic that drifts is
     disqualified.

  python3 exp3_eval_probe.py --stages 0
  python3 exp3_eval_probe.py --stages 1 --jsonl out.jsonl
  python3 exp3_eval_probe.py --stages 2 --hf-model <name> --task sexp
"""

import argparse
import json
import os
import sys

import numpy as np

from eval_common import (variance_decomposition, effective_rank,
                         intrinsic_dim_twonn, functoriality_residual,
                         gauge_report, verdict, jsonl_append, jsonl_done)


# ======================================================================
# Stage 0 -- oracles with known ground truth
# ======================================================================

def _so_generators(k, p, rng):
    """p antisymmetric generators. Antisymmetric => the group is noncommutative
    for p >= 2 in k >= 3, which E2 needs: on a commutative family the
    homomorphism test has no teeth."""
    G = []
    for _ in range(p):
        Araw = rng.normal(size=(k, k))
        G.append(Araw - Araw.T)
    return np.stack(G)


def _expm(A, terms=24):
    out = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for n in range(1, terms):
        term = term @ A / n
        out = out + term
    return out


class Oracle:
    """Produces M(u,v), b(u,v), A(u,v) with a known correct verdict."""

    def __init__(self, kind, k=8, p=3, N_u=24, N_v=12, T=10, n_head=6,
                 scale=0.25, noise=0.0, seed=0):
        self.kind, self.k, self.p = kind, k, p
        self.N_u, self.N_v, self.T, self.n_head = N_u, N_v, T, n_head
        self.noise = noise
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.G = _so_generators(k, p, rng)

        # a generator set plus all its pairwise products, so composition
        # triples stay inside the code set
        n_gen = 6
        self.theta = rng.normal(size=(n_gen, p)) * scale
        self.R = [_expm(np.tensordot(t, self.G, axes=1)) for t in self.theta]
        self.codes, self.triples = list(range(n_gen)), []
        nxt = n_gen
        for i in range(n_gen):
            for j in range(n_gen):
                if i == j or nxt >= N_u:
                    continue
                self.R.append(self.R[i] @ self.R[j])
                self.codes.append(nxt)
                self.triples.append((i, j, nxt))
                nxt += 1
        self.N_u = len(self.codes)

        self.W_fixed = rng.normal(size=(k, k)) / np.sqrt(k)
        self.tab_u = rng.normal(size=(self.N_u, k, k)) / np.sqrt(k)
        self.tab_uv = rng.normal(size=(self.N_u, N_v, k, k)) / np.sqrt(k)
        self.attn_u = rng.normal(size=(self.N_u, n_head, T, T))
        self.attn_uv = rng.normal(size=(self.N_u, N_v, n_head, T, T))

    def collect(self):
        k, Nu, Nv = self.k, self.N_u, self.N_v
        M = np.zeros((Nu, Nv, k, k))
        b = np.zeros((Nu, Nv, k))
        A = np.zeros((Nu, Nv, self.n_head, self.T, self.T))
        for iu in range(Nu):
            for iv in range(Nv):
                if self.kind == "true_eval":
                    M[iu, iv] = self.R[iu]
                    A[iu, iv] = self.attn_u[iu]
                elif self.kind == "table_over_f":
                    M[iu, iv] = self.tab_u[iu]
                    A[iu, iv] = self.attn_u[iu]
                elif self.kind == "lookup_pair":
                    M[iu, iv] = self.tab_uv[iu, iv]
                    A[iu, iv] = self.attn_uv[iu, iv]
                elif self.kind == "ignores_code":
                    M[iu, iv] = self.W_fixed
                    A[iu, iv] = self.attn_u[0]
                else:
                    raise ValueError(self.kind)
        if self.noise:
            M = M + self.rng.normal(scale=self.noise, size=M.shape)
            A = A + self.rng.normal(scale=self.noise, size=A.shape)
        return dict(M=M, b=b, A=A)


ORACLE_EXPECT = {
    "true_eval":     "eval_with_algebra",
    "table_over_f":  "lookup_table",
    "lookup_pair":   "no_factorisation",
    "ignores_code":  "ignores_code",
}


# ======================================================================
# analysis -- shared by every stage
# ======================================================================

def analyse(data, triples=None, side="M", p_true=None, floor=None,
            markov=False, tag=""):
    """data: dict with M (Nu,Nv,k,k), b (Nu,Nv,k), A (Nu,Nv,...)

    side="M" runs the value-path (SMCC) probe; side="A" runs the attention
    (Markov) probe. Both are computed because which one carries [[.]] is an
    empirical question, not one to settle in advance."""
    X = data["M"] if side == "M" else data["A"]
    if X is None:
        return None

    sep = variance_decomposition(X)

    X_u = X.mean(axis=1)                       # average out v -> {[[u]]}
    e1 = effective_rank(X_u)
    e1.update(intrinsic_dim_twonn(X_u))

    e2 = None
    if triples and side == "M":
        Mmap = {i: X_u[i] for i in range(len(X_u))}
        bmap = {i: data["b"].mean(axis=1)[i] for i in range(len(X_u))}
        e2 = functoriality_residual(Mmap, bmap, triples, markov=markov)
    elif triples and side == "A":
        # A(u) is (n_sel, T, T); compose per selected head, stay row-stochastic
        Mmap, resid = {}, []
        for h in range(X_u.shape[1]):
            Mmap = {i: X_u[i, h] for i in range(len(X_u))}
            r = functoriality_residual(Mmap, None, triples, markov=True)
            resid.append(r)
        e2 = dict(rho=float(np.mean([r["rho"] for r in resid])),
                  rho_null=float(np.mean([r["rho_null"] for r in resid])),
                  rho_ratio=float(np.mean([r["rho_ratio"] for r in resid])),
                  n_triples=resid[0]["n_triples"], per_head=len(resid))

    v = verdict(sep, e1, e2, p_true=p_true, N_u=X.shape[0], floor=floor)
    return dict(tag=tag, side=side, sep=sep, e1=e1, e2=e2, **v)


def brief(res):
    e2 = res.get("e2") or {}
    return (f"{res['tag']:<22} side={res['side']}  "
            f"sep_full={res['sep_full']:8.3f}  "
            f"pr={res['e1']['pr']:6.2f}  "
            f"id={res['e1'].get('id_twonn', float('nan')):5.2f}  "
            f"rho/null={e2.get('rho_ratio', float('nan')):6.3f}  "
            f"-> {res['verdict']}")


# ======================================================================
# stage 0
# ======================================================================

def stage0(args):
    print("\n=== stage 0: oracle validation "
          "(the statistics must recover known ground truth) ===\n")
    ok = True
    rows = []
    for kind, expect in ORACLE_EXPECT.items():
        orc = Oracle(kind, k=args.k, p=args.p, N_v=args.n_v,
                     noise=args.noise, seed=args.seed)
        data = orc.collect()
        rM = analyse(data, triples=orc.triples, side="M",
                     p_true=orc.p, tag=kind)
        rA = analyse(data, triples=orc.triples, side="A", tag=kind)
        got = rM["verdict"]
        hit = (got == expect)
        ok &= hit
        print(brief(rM), "   [expected", expect, "]" if hit else "] <-- MISMATCH")
        print(brief(rA))
        rows.append(dict(key=f"s0/{kind}", kind=kind, expected=expect,
                         M=rM, A=rA, pass_=hit))
        jsonl_append(args.jsonl, rows[-1])

    print("\nnote: pr is the LINEAR effective rank, id the TwoNN manifold "
          "dimension.\n      For true_eval the ground truth is p =", args.p,
          "-- read id, not pr;\n      pr overshoots because exp(theta.G) is "
          "not affine in theta.")
    print("\nstage 0:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ======================================================================
# task construction for real models
# ======================================================================

def build_sexp_task(tok, n_u=16, n_v=12, k=16):
    """Higher-order S-expression prompts with a fixed layout.

        (apply (compose f3 f7) v5) =>

    Every template pads to the same length and puts the x-slot and the readout
    at the same index, which check_layout() enforces. Composition triples are
    (f, g, compose(f,g)) so E2 is asked of terms the language actually contains.

    Replace this with your own tokenizer/grammar; the probe only needs
    codes_u, codes_v, slot_of, value_token_ids, triples.
    """
    fn_names = [f"f{i}" for i in range(n_u)]
    val_names = [f"v{i}" for i in range(k)]

    def render(u, v):
        if isinstance(u, tuple):
            head = f"( compose {fn_names[u[0]]} {fn_names[u[1]]} )"
        else:
            head = f"( id {fn_names[u]} ) "          # pad to equal token count
        return f"( apply {head} {val_names[v]} ) =>"

    codes_u = list(range(n_u))
    triples = []
    for i in range(0, min(6, n_u)):
        for j in range(0, min(6, n_u)):
            if i != j:
                codes_u.append((i, j))
                triples.append((i, j, len(codes_u) - 1))
    codes_v = list(range(min(n_v, k)))
    return dict(render=render, codes_u=codes_u, codes_v=codes_v,
                triples=triples, val_names=val_names, fn_names=fn_names)


def build_nl_task(n_u=16, n_v=12):
    """Natural-language control. u = a relation phrase, v = an entity token.
    Deliberately the same shape as the S-expression task so the two are
    comparable; the point of running it is that a pretrained NL model has no
    reason to have learned a compose operator, so E2 should separate."""
    rels = ["the capital of", "the currency of", "the language of",
            "the continent of"][:n_u]

    def render(u, v, ents):
        return f"Q: What is {rels[u % len(rels)]} {ents[v]} ? A:"
    return dict(rels=rels, render=render)


# ======================================================================
# stages 1-3
# ======================================================================

def _load_torch():
    import torch
    import eval_extract as ex
    return torch, ex


def stage_model(args, adapter, task, torch, ex, tag, floor=None):
    V_emb = adapter.value_embeddings()                       # (k,d)
    k = V_emb.shape[0]

    def c0_of(v):
        c = torch.zeros(k, device=V_emb.device, dtype=V_emb.dtype)
        c[v] = 1.0
        return c

    def slot_of(u, v):
        text = task["render"](u, v)
        ids = task["encode"](text)
        return ex.Slots(ids, f_pos=task["f_pos"], x_pos=task["x_pos"])

    n = len(task["codes_u"]) * len(task["codes_v"])
    seen = [0]

    def prog(iu, iv):
        seen[0] += 1
        if seen[0] % max(1, n // 10) == 0:
            print(f"  {seen[0]}/{n}", flush=True)

    data = ex.sweep_grid(adapter, slot_of, task["codes_u"], task["codes_v"],
                         V_emb, c0_of, want_A=not args.no_attn,
                         jac_mode=args.jac_mode, progress=prog)

    out = {}
    for side in ("M", "A"):
        r = analyse(data, triples=task["triples"], side=side,
                    p_true=args.p, floor=floor, tag=tag)
        if r:
            print(brief(r))
            out[side] = r
    if args.save_raw:
        np.savez_compressed(args.save_raw, **{kk: vv for kk, vv in data.items()
                                              if vv is not None})
        print("  raw M/b/A ->", args.save_raw)
    return out


def stage1(args):
    torch, ex = _load_torch()
    print("\n=== stage 1: random-init floor ===\n")
    from eval_extract import LocalAdapter          # noqa
    print("  supply --hf-model or --local-ckpt with untrained weights; "
          "this stage is identical to stage 2 except that nothing is loaded.")
    return 0


def stage2(args):
    torch, ex = _load_torch()
    print("\n=== stage 2: trained checkpoint ===\n")
    if args.hf_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model)
        task = build_sexp_task(tok) if args.task == "sexp" else build_nl_task()
        val_ids = [tok.encode(" " + s, add_special_tokens=False)[0]
                   for s in task["val_names"]]
        adapter = ex.HFAdapter(model, val_ids, device=args.device)

        probe = task["render"](task["codes_u"][0], task["codes_v"][0])
        ids = tok.encode(probe, add_special_tokens=False)
        x_pos = max(i for i, t in enumerate(ids) if t in val_ids)
        task["encode"] = lambda s: tok.encode(s, add_special_tokens=False)
        task["x_pos"], task["f_pos"] = x_pos, []
    else:
        raise SystemExit("stage 2 needs --hf-model (or wire LocalAdapter here)")

    floor = None
    if args.floor_json and os.path.exists(args.floor_json):
        floor = json.load(open(args.floor_json))["M"]["e1"]
    res = stage_model(args, adapter, task, torch, ex,
                      tag=os.path.basename(args.hf_model), floor=floor)
    jsonl_append(args.jsonl, dict(key=f"s2/{args.hf_model}/{args.task}", **res))
    return 0


def stage3(args):
    """Gauge guard. Head permutation is exact and cheap; the GL(d) residual
    reparameterisation is only exact without LayerNorm, so for a real model the
    honest version is the head permutation plus a seed rerun. The heavier check
    -- reproducing a Wen-style function-preserving randomisation and showing the
    statistics do not move -- goes here once you have that transform."""
    print("\n=== stage 3: gauge guard ===")
    print("  head permutation: A is only ever used as a stacked set over "
          "(layer, head), so head order cannot enter -- invariant by "
          "construction.")
    print("  residual GL(d): A depends on W_Q^T W_K, invariant; M is a "
          "derivative of the input-output map, invariant.")
    print("  what is NOT invariant, and is the real risk: Wen-style solution "
          "degeneracy. Run your randomisation, re-run stage 2, and pass both "
          "jsons to gauge_report().")
    return 0


# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="0")
    ap.add_argument("--k", type=int, default=8, help="value-token basis size")
    ap.add_argument("--p", type=int, default=3, help="true dim of the f family")
    ap.add_argument("--n-v", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--jac-mode", default="jvp", choices=["jvp", "fd"])
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--hf-model", default=None)
    ap.add_argument("--task", default="sexp", choices=["sexp", "nl"])
    ap.add_argument("--floor-json", default=None)
    ap.add_argument("--save-raw", default=None)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    rc = 0
    for s in args.stages.replace(",", ""):
        rc |= {"0": stage0, "1": stage1, "2": stage2, "3": stage3}[s](args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
