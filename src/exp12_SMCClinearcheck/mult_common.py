"""
mult_common.py
=====================================================================
Shared infrastructure for the two falsification experiments on the
"Transformer = SMCC (linear, no !) x Markov category" thesis.

CORE TASK -- multiplicative multiplicity, orthogonal to everything else
---------------------------------------------------------------------
Sequence (fixed length, independent of m):

    [BOS] [X:x] [A:a_1] ... [A:a_K] [Q]

with x ~ U[-1,1] and a_i ~ U[-1,1], both injected as CONTINUOUS scalars
through learned linear maps into the residual stream.

    target_m(x, a) = ( sum_i a_i / sqrt(K) ) * x^m / sigma_m
    sigma_m = 1/sqrt(2m+1)   (so Var[x^m/sigma_m] = 1 for x ~ U[-1,1])

The ONLY thing that varies with m is the multiplicative multiplicity of
the designated variable x. Sequence length, token count, the additive
reduction over the a_i, and the target variance are all held fixed.

  m = 1  ->  x is used K times ADDITIVELY.  This is the (+)-copy /
            biproduct diagonal that Vect has for free, and that
            frozen-softmax attention Z = AV implements natively.
  m >= 2 ->  x must be duplicated MULTIPLICATIVELY (x (x) x), i.e. the
            (x)-diagonal that a symmetric monoidal closed category
            without ! does NOT have.

So m* (the largest m a given architecture can fit) is a direct empirical
measure of the available contraction budget.  Its scaling decides:

    m* ~ L,  flat in d      -> depth-graded !_L  (paper's claim C)
    m* ~ d,  flat in L      -> width-graded !_d  (Tracr-style; claim C falsified)
    m* unbounded            -> unrestricted contraction (claim B falsified)

WHY CONTINUOUS x IS ESSENTIAL
---------------------------------------------------------------------
If x were a discrete token from a small vocabulary, x^m would be a
lookup table of |V| entries and contraction would cost nothing --- the
resource-linearity claim is VACUOUS for enumerable values.  Any
falsification test must therefore use non-enumerable (continuous,
high-entropy) values.  This is a substantive design constraint, not an
implementation detail.

WHY GELU AND NOT RELU
---------------------------------------------------------------------
Experiment 1's probe measures polynomial degree.  A ReLU network is
piecewise linear, so all local higher derivatives vanish a.e. even
though the global function is not degree 1.  We default to GELU (smooth)
and use a GLOBAL Chebyshev spectrum rather than a local Taylor jet, so
the probe stays meaningful for either activation.

CPU / GPU
---------------------------------------------------------------------
One code path.  --preset selects scale; --device auto-detects.  Batches
are generated synthetically on-device (no DataLoader), so GPU scaling is
just a matter of larger --preset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

@dataclass
class ModelCfg:
    n_layer: int = 2
    d_model: int = 64
    n_head: int = 4
    d_ff_mult: int = 4
    K: int = 8                 # number of a_i tokens (fixed across the m sweep)
    act: str = "gelu"          # gelu | relu | tanh
    use_layernorm: bool = True

    @property
    def seq_len(self) -> int:
        return 3 + self.K      # BOS, X, K x A, Q

    def n_params(self) -> int:
        d, L, f = self.d_model, self.n_layer, self.d_model * self.d_ff_mult
        per_layer = 4 * d * d + 2 * d * f + 2 * d  # attn qkvo + mlp + 2 LN
        return L * per_layer + 4 * d + self.seq_len * d + d + 1


@dataclass
class TrainCfg:
    steps: int = 2000
    batch: int = 256
    lr: float = 3e-3
    warmup: int = 100
    wd: float = 0.0
    eval_batch: int = 4096
    eval_every: int = 0        # 0 -> only at the end
    seed: int = 0
    amp: bool = False          # bf16 autocast (GPU only)
    compile: bool = False
    ood_ratio: float = 2.197   # held-constant overshoot r^m for the matched band


PRESETS = {
    # tiny: correctness smoke test, seconds on one CPU core
    "cpu_smoke": dict(
        m_list=[1, 2, 3], L_list=[1, 2], d_list=[32, 64], seeds=1,
        steps=300, batch=128, eval_batch=2048, K=8,
    ),
    # the real small-scale run: minutes-to-an-hour on CPU
    "cpu_small": dict(
        m_list=[1, 2, 3, 4, 5, 6], L_list=[1, 2, 3], d_list=[32, 64, 128], seeds=2,
        steps=3000, batch=256, eval_batch=8192, K=8,
    ),
    # full scale, one GPU
    "gpu_full": dict(
        m_list=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], L_list=[1, 2, 3, 4, 6, 8],
        d_list=[64, 128, 256, 512], seeds=3,
        steps=30000, batch=1024, eval_batch=65536, K=8, amp=True,
    ),
    # decisive-but-cheap GPU run: one seed, coarser grid, still 10x the CPU steps
    "gpu_fast": dict(
        m_list=[1, 2, 3, 4, 5, 6, 8, 10], L_list=[1, 2, 4, 8],
        d_list=[64, 128, 256], seeds=1,
        steps=10000, batch=1024, eval_batch=65536, K=8, amp=True,
    ),
    # iso-parameter control: L*d^2 held ~constant, isolates depth from capacity
    "gpu_iso": dict(
        m_list=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], L_list=[1, 2, 4, 8], d_list=None,
        seeds=3, steps=30000, batch=1024, eval_batch=65536, K=8, amp=True,
        iso_budget=8 * 128 * 128,   # L * d^2 target
    ),
}


def output_root() -> str:
    """
    Vertex AI Custom Jobs export AIP_MODEL_DIR (a gs:// or /gcs/ path).  Local
    disk on a Spot/preemptible worker is wiped on restart, so anything that must
    survive a preemption has to end up under a GCS-backed path.
    """
    return os.environ.get("AIP_MODEL_DIR") or os.environ.get("OUT_DIR") or "./results"


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iso_widths(L_list, budget, n_head=4):
    """d such that L*d^2 ~ budget, snapped to a multiple of n_head."""
    out = {}
    for L in L_list:
        d = int(round(math.sqrt(budget / L) / n_head)) * n_head
        out[L] = max(n_head * 4, d)
    return out


# ---------------------------------------------------------------------
# task
# ---------------------------------------------------------------------

def sigma_pow(m: int) -> float:
    """std of x^m for x ~ U[-1,1]."""
    return 1.0 / math.sqrt(2 * m + 1)


def ood_hi(m: int, ratio: float = 2.197) -> float:
    """
    Scale-MATCHED extrapolation band.

    A fixed band such as x in [-1.3, 1.3] is not comparable across m: the
    target's value range overshoots the training support by r^m, so at r=1.3
    the overshoot is 1.3x for m=1 but 4.8x for m=6.  Extrapolation therefore
    gets intrinsically harder with m for reasons that have nothing to do with
    contraction, which biases m* downward and spuriously favours the
    "bounded contraction budget" hypothesis.

    Fixing r^m = ratio instead holds the overshoot constant:
        r_m = ratio ** (1/m)
    ratio = 2.197 = 1.3^3 reproduces the old band exactly at m = 3, so the
    two metrics are directly comparable there.
    """
    return float(ratio ** (1.0 / m))


def make_batch(B: int, K: int, m: int, device, generator=None,
               x_lo: float = -1.0, x_hi: float = 1.0, dtype=torch.float32):
    """Returns (x, a, y).  x: (B,)  a: (B,K)  y: (B,) with Var[y] ~ 1."""
    x = torch.empty(B, device=device, dtype=dtype).uniform_(x_lo, x_hi, generator=generator)
    a = torch.empty(B, K, device=device, dtype=dtype).uniform_(-1.0, 1.0, generator=generator)
    coef = a.sum(dim=1) / math.sqrt(K)              # Var ~ 1/3
    y = coef * (x.pow(m) / sigma_pow(m))
    return x, a, y


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------

ACTS = {"gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}

# token type ids
TOK_BOS, TOK_X, TOK_A, TOK_Q = 0, 1, 2, 3


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        d, f = cfg.d_model, cfg.d_model * cfg.d_ff_mult
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, f)
        self.fc2 = nn.Linear(f, d)
        self.act = ACTS[cfg.act]

    def forward(self, h):
        B, T, d = h.shape
        z = self.ln1(h)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        shp = (B, T, self.n_head, d // self.n_head)
        q, k, v = (t.view(shp).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)          # softmax: Markov kernel
        o = o.transpose(1, 2).reshape(B, T, d)
        h = h + self.proj(o)                                 # residual: (+)-copy
        z = self.ln2(h)
        h = h + self.fc2(self.act(self.fc1(z)))              # MLP: cartesian realization
        return h


class MultTransformer(nn.Module):
    """Encoder-style transformer with continuous scalar injection at X and A slots."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(4, d)
        self.pos = nn.Embedding(cfg.seq_len, d)
        self.inj_x = nn.Linear(1, d, bias=False)   # continuous value -> residual stream
        self.inj_a = nn.Linear(1, d, bias=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(d) if cfg.use_layernorm else nn.Identity()
        self.head = nn.Linear(d, 1)
        self.apply(self._init)

    @staticmethod
    def _init(mod):
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)

    # -- split embed / forward so Experiment 1 can perturb the X slot directly --

    def embed(self, x, a):
        """x: (B,) a: (B,K) -> e: (B,T,d).  X slot index is 1."""
        B, K = a.shape
        T = self.cfg.seq_len
        ids = torch.full((B, T), TOK_A, device=a.device, dtype=torch.long)
        ids[:, 0] = TOK_BOS
        ids[:, 1] = TOK_X
        ids[:, -1] = TOK_Q
        e = self.tok(ids) + self.pos(torch.arange(T, device=a.device))[None]
        e = e.clone()
        e[:, 1] = e[:, 1] + self.inj_x(x[:, None].to(e.dtype))
        e[:, 2:2 + K] = e[:, 2:2 + K] + self.inj_a(a[..., None].to(e.dtype))
        return e

    def forward_from_embed(self, e):
        h = e
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h[:, -1])).squeeze(-1)

    def forward(self, x, a):
        return self.forward_from_embed(self.embed(x, a))

    @property
    def x_slot(self) -> int:
        return 1


# ---------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------

def r2_score(pred, y):
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


@torch.no_grad()
def evaluate(model, m, cfg: ModelCfg, device, n=8192, batch=4096, seed=1234,
             x_lo=-1.0, x_hi=1.0):
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    preds, ys = [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        x, a, y = make_batch(b, cfg.K, m, device, g, x_lo, x_hi)
        preds.append(model(x, a).float())
        ys.append(y.float())
        done += b
    return r2_score(torch.cat(preds), torch.cat(ys))


def train_one(mcfg: ModelCfg, tcfg: TrainCfg, m: int, device, verbose=False):
    torch.manual_seed(tcfg.seed)
    model = MultTransformer(mcfg).to(device)
    if tcfg.compile and device.type == "cuda":
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.wd,
                            betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, tcfg.warmup))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / tcfg.steps))))
    g = torch.Generator(device=device).manual_seed(tcfg.seed + 7)
    use_amp = tcfg.amp and device.type == "cuda"

    t0 = time.time()
    model.train()
    for step in range(tcfg.steps):
        x, a, y = make_batch(tcfg.batch, mcfg.K, m, device, g)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = F.mse_loss(model(x, a), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and tcfg.eval_every and (step + 1) % tcfg.eval_every == 0:
            print(f"    step {step+1:6d}  loss {loss.item():.4f}", flush=True)

    r2_id = evaluate(model, m, mcfg, device, n=tcfg.eval_batch)
    r2_ood = evaluate(model, m, mcfg, device, n=tcfg.eval_batch, seed=99,
                      x_lo=-1.3, x_hi=1.3)                       # fixed band (legacy)
    hi = ood_hi(m, tcfg.ood_ratio)
    r2_ood_m = evaluate(model, m, mcfg, device, n=tcfg.eval_batch, seed=99,
                        x_lo=-hi, x_hi=hi)                       # scale-matched band
    return model, dict(r2=r2_id, r2_ood=r2_ood, r2_ood_matched=r2_ood_m,
                       ood_hi=round(hi, 4), final_loss=loss.item(),
                       wall_s=round(time.time() - t0, 2))


# ---------------------------------------------------------------------
# Experiment 1 probes: effective polynomial degree
# ---------------------------------------------------------------------

def chebyshev_spectrum(vals: torch.Tensor, nodes: int) -> torch.Tensor:
    """
    Discrete Chebyshev transform of a function sampled at Chebyshev-Gauss
    nodes t_j = cos(pi (j+1/2)/N) on [-1,1].
    vals: (..., N) -> coeffs: (..., N)
    A degree-p polynomial has exactly zero coefficients above index p.
    """
    N = nodes
    j = torch.arange(N, dtype=vals.dtype, device=vals.device)
    k = torch.arange(N, dtype=vals.dtype, device=vals.device)
    # DCT-II matrix
    M = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / N)     # (N,N)
    c = (2.0 / N) * torch.einsum("...j,kj->...k", vals, M)
    c[..., 0] = c[..., 0] / 2
    return c


TOLS = (0.3, 0.1, 0.03, 0.01, 0.003, 1e-4, 1e-6)


def truncation_curve(coeffs: torch.Tensor, pool: bool = True) -> torch.Tensor:
    """
    tail[p] = || c_{>p} || / || c_{>=1} ||   (constant term excluded).
    An exact degree-p polynomial gives tail[p] == 0.

    pool=True aggregates coefficient ENERGY across the batch before taking the
    ratio.  This matters: contexts where sum_i a_i happens to be near zero
    carry almost no x-dependence, so a per-context degree there is pure noise.
    Averaging per-context degrees lets those contexts dominate; pooling energy
    weights each context by how much signal it actually carries.
    """
    c = coeffs[..., 1:]
    energy = c.pow(2)
    if pool:
        energy = energy.reshape(-1, energy.shape[-1]).mean(0)
    total = energy.sum(-1, keepdim=True).clamp_min(1e-300)
    tail = energy.flip(-1).cumsum(-1).flip(-1) / total       # tail[p] = ||c_{>p}||^2 / total
    return tail.clamp_min(0).sqrt()                          # index p == degree p


def degree_from_curve(tail: torch.Tensor, eps: float) -> float:
    ok = tail < eps
    return float(ok.float().argmax(-1).item()) if bool(ok.any()) else float(tail.shape[-1])


def effective_degree(coeffs: torch.Tensor, eps: float = 1e-2, pool: bool = True):
    return degree_from_curve(truncation_curve(coeffs, pool), eps)


def degrees_at_tols(coeffs: torch.Tensor, tols=TOLS):
    """dict tol -> effective degree (energy-pooled), plus the truncation curve."""
    tail = truncation_curve(coeffs, pool=True)
    out = {f"deg@{t:g}": degree_from_curve(tail, t) for t in tols}
    out["curve"] = tail.cpu().tolist()
    return out


@torch.no_grad()
def probe_input_degree(model, mcfg: ModelCfg, device, n_ctx=64, nodes=32,
                       eps=1e-2, seed=5, dtype=torch.float64):
    """
    P-in: hold the context (a_1..a_K) fixed, sweep the ACTUAL scalar x over
    Chebyshev nodes in [-1,1], and read off the Chebyshev degree of the
    model's output as a function of x.  Ground truth for task m is exactly m.
    """
    model = model.to(dtype).eval()
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.empty(n_ctx, mcfg.K, device=device, dtype=dtype).uniform_(-1, 1, generator=g)
    j = torch.arange(nodes, device=device, dtype=dtype)
    t = torch.cos(math.pi * (j + 0.5) / nodes)                       # (N,)
    xs = t[None, :].expand(n_ctx, nodes).reshape(-1)                 # (n_ctx*N,)
    aa = a[:, None, :].expand(n_ctx, nodes, mcfg.K).reshape(-1, mcfg.K)
    out = model(xs, aa).reshape(n_ctx, nodes)
    c = chebyshev_spectrum(out, nodes)
    res = degrees_at_tols(c)
    res["spectrum"] = c.abs().mean(0).cpu().tolist()
    model.float()
    return res


@torch.no_grad()
def probe_embedding_degree(model, mcfg: ModelCfg, device, n_dir=16, n_ctx=8,
                           nodes=32, rho=1.0, eps=1e-2, seed=11,
                           dtype=torch.float64):
    """
    P-emb: perturb the X-slot residual vector along random directions of norm
    rho*||base||, sweep the coefficient over Chebyshev nodes, and read off the
    degree.  This measures the multiplicative capacity the architecture makes
    available in directions the task does not exercise -- i.e. the ceiling,
    not the used budget.
    """
    model = model.to(dtype).eval()
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty(n_ctx, device=device, dtype=dtype).uniform_(-1, 1, generator=g)
    a = torch.empty(n_ctx, mcfg.K, device=device, dtype=dtype).uniform_(-1, 1, generator=g)
    e0 = model.embed(x, a)                                            # (n_ctx,T,d)
    base = e0[:, model.x_slot, :]
    scale = base.norm(dim=-1, keepdim=True).clamp_min(1e-6) * rho

    j = torch.arange(nodes, device=device, dtype=dtype)
    t = torch.cos(math.pi * (j + 0.5) / nodes)

    cs = []
    for _ in range(n_dir):
        u = torch.randn(base.shape, generator=g, device=device, dtype=dtype)
        u = u / u.norm(dim=-1, keepdim=True) * scale                  # (n_ctx,d)
        e = e0[:, None].expand(n_ctx, nodes, *e0.shape[1:]).clone()
        e[:, :, model.x_slot, :] = base[:, None, :] + t[None, :, None] * u[:, None, :]
        out = model.forward_from_embed(e.reshape(n_ctx * nodes, *e0.shape[1:]))
        out = out.reshape(n_ctx, nodes)
        cs.append(chebyshev_spectrum(out, nodes))
    c = torch.cat(cs, 0)
    res = degrees_at_tols(c)
    res["spectrum"] = c.abs().mean(0).cpu().tolist()
    model.float()
    return res


# ---------------------------------------------------------------------
# CLI plumbing shared by both experiments
# ---------------------------------------------------------------------

def base_argparser(desc: str):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--preset", default="cpu_smoke", choices=list(PRESETS))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--act", default="gelu", choices=list(ACTS))
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--m_list", type=int, nargs="*", default=None)
    p.add_argument("--L_list", type=int, nargs="*", default=None)
    p.add_argument("--d_list", type=int, nargs="*", default=None)
    p.add_argument("--eps", type=float, default=1e-2, help="degree-probe tolerance")
    p.add_argument("--threshold", type=float, default=0.9, help="R^2 success threshold")
    p.add_argument("--jsonl", default=None, help="append records here (resumable chunked runs)")
    p.add_argument("--stages", default="012", help="which exp1 stages to run")
    p.add_argument("--shard", default="0/1",
                   help="k/N -- run only configs with index %% N == k. These models are "
                        "tiny (seq_len 11, ~1e6 params), so one process leaves the GPU "
                        "almost idle; run 8-16 shards concurrently per device.")
    p.add_argument("--light_save", action="store_true",
                   help="skip rewriting the full JSON each step (the jsonl is the record)")
    p.add_argument("--ood_ratio", type=float, default=2.197,
                   help="held-constant overshoot r^m for the scale-matched OOD band")
    return p


def resolve_preset(args):
    cfg = dict(PRESETS[args.preset])
    for k in ("steps", "batch", "seeds", "m_list", "L_list", "d_list"):
        v = getattr(args, k, None)
        if v:
            cfg[k] = v
    if cfg.get("d_list") is None:
        cfg["d_map"] = iso_widths(cfg["L_list"], cfg["iso_budget"], args.n_head)
    return cfg


def append_jsonl(rec, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def parse_shard(spec: str):
    k, n = spec.split("/")
    k, n = int(k), int(n)
    assert 0 <= k < n, f"bad shard {spec}"
    return k, n


def save(records, path, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(meta=meta or {}, records=records), f, indent=1)
    print(f"[saved] {path}  ({len(records)} records)")
