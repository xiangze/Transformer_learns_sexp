"""
three_arch_curriculum.py — production three-architecture boundary comparison
=====================================================================
Does the state-tracking failure boundary n*(L) distinguish architectures by
their categorical forward structure?

    Transformer (depth L):  predicted  n* ~ c * 2^L   (associative scan =
                            log-grade !_{log n})  -> EXPONENTIAL in L
    RNN (1 layer):          predicted  n* unbounded   (true !)  -> FLAT-high
    diagonal SSM:           predicted  n* small, flat (TC0)      -> FLAT-low

Why this file exists: the earlier direct-training run made ALL archs collapse at
n=8, which was a TRAINABILITY confound (rnn_calib.py showed a 1-layer GRU reaches
n=32 WITH CURRICULUM). So here EVERY architecture is trained with the SAME
curriculum (grow n: 2->4->8->..., warm-starting one model), so a failure at a
given n is an EXPRESSIVITY/architecture limit, not an optimisation artifact.

For each (kind, L) we curriculum-train up the n-ladder, recording the largest n
mastered (acc>=thresh). That ladder top is n*(L). We then fit n*(L) across L to
linear vs exponential (per architecture) and print the verdict.

Task: explicit permutation composition over S_V (same as three_arch.py):
    x0 , g_{i1} .. g_{in}  ->  (g_in o .. o g_i1)(x0),  scored on the final state.

Usage
    python three_arch_curriculum.py --V 5 --Ls 1 2 3 4 \
        --ns 2 3 4 6 8 12 16 24 32 48 64 --kinds tf rnn ssm --seeds 3
Resumable via JSON checkpoint. GPU auto-detected; bump --d, --ns, --steps there.
"""

import math, json, os, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
def perm_gens(V, n_gens, device, seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(V, generator=g) for _ in range(n_gens)]).to(device)


def make_batch(B, V, n, gens, device):
    G = gens.shape[0]
    T = 1 + n + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    x0 = torch.randint(0, V, (B,), device=device)
    toks[:, 0] = x0
    idx = torch.randint(0, G, (B, n), device=device)
    toks[:, 1:1 + n] = idx + V
    state = x0.clone()
    for j in range(n):
        state = torch.gather(gens[idx[:, j]], 1, state[:, None]).squeeze(1)
    toks[:, T - 1] = state
    return toks, T - 1


# ------------------------------------------------------------------ models
class TF(nn.Module):
    def __init__(self, ntok, d, h, L, maxT):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(maxT, d)
        self.h, self.d = h, d
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), ln2=nn.LayerNorm(d),
            qkv=nn.Linear(d, 3 * d, bias=False), o=nn.Linear(d, d, bias=False),
            f1=nn.Linear(d, 4 * d), f2=nn.Linear(4 * d, d))) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, ntok)

    def forward(self, t):
        B, T = t.shape
        x = self.tok(t) + self.pos(torch.arange(T, device=t.device))[None]
        dh = self.d // self.h
        for b in self.blocks:
            z = b["ln1"](x); q, k, v = b["qkv"](z).chunk(3, -1)
            q, k, v = [y.view(B, T, self.h, dh).transpose(1, 2) for y in (q, k, v)]
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + b["o"](o.transpose(1, 2).reshape(B, T, self.d))
            x = x + b["f2"](F.gelu(b["f1"](b["ln2"](x))))
        return self.head(self.lnf(x))


class RNN(nn.Module):
    def __init__(self, ntok, d, L, cell="gru"):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](d, d, num_layers=L,
                                                          batch_first=True)
        self.head = nn.Linear(d, ntok)

    def forward(self, t):
        y, _ = self.rnn(self.tok(t)); return self.head(y)


class DiagSSM(nn.Module):
    def __init__(self, ntok, d, L):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        self.a = nn.ParameterList([nn.Parameter(torch.rand(d) * 0.5 + 0.4) for _ in range(L)])
        self.Bp = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.Cp = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.head = nn.Linear(d, ntok)

    def forward(self, t):
        x = self.tok(t); B, T, d = x.shape
        for a, Bp, Cp, ln in zip(self.a, self.Bp, self.Cp, self.ln):
            u = Bp(ln(x)); a_ = a.clamp(0.01, 0.999)
            h = torch.zeros(B, d, device=x.device); outs = []
            for j in range(T):
                h = a_ * h + u[:, j]; outs.append(h)
            x = x + Cp(torch.stack(outs, 1))
        return self.head(x)


def build(kind, ntok, d, L, maxT, h=4):
    if kind == "tf":  return TF(ntok, d, h, L, maxT)
    if kind == "rnn": return RNN(ntok, d, L)
    if kind == "ssm": return DiagSSM(ntok, d, L)


@torch.no_grad()
def evaluate(m, V, n, gens, device, B=2048):
    m.eval()
    toks, ap = make_batch(B, V, n, gens, device)
    return (m(toks)[:, ap - 1, :].argmax(-1) == toks[:, ap]).float().mean().item()


# ------------------------------------------------------------------ shared curriculum
def curriculum_boundary(kind, V, gens, d, L, ns, device, steps=2000, B=256,
                        lr=2e-3, seed=0, thresh=0.95, h=4, amp=True):
    """Train ONE model up the n-ladder (warm-start). Return {n: acc} and the
    largest n mastered (acc>=thresh). Stopping: if a stage stalls (acc<0.5 after
    the step budget), stop — that n is beyond this (kind,L)'s reach."""
    torch.manual_seed(seed)
    ntok = V + gens.shape[0]
    maxT = max(ns) + 4
    m = build(kind, ntok, d, L, maxT, h).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    hist = {}
    for n in ns:
        m.train()
        best = 0.0
        for step in range(steps):
            toks, ap = make_batch(B, V, n, gens, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = F.cross_entropy(m(toks)[:, ap - 1, :].float(), toks[:, ap])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if (step + 1) % 500 == 0:
                a = evaluate(m, V, n, gens, device)
                best = max(best, a)
                if a >= 0.98:
                    break
                m.train()
        acc = max(best, evaluate(m, V, n, gens, device))
        hist[n] = acc
        if acc < 0.5:            # stalled: this n is beyond reach for (kind,L)
            break
    reached = max([n for n, a in hist.items() if a >= thresh], default=0)
    return hist, reached


# ------------------------------------------------------------------ boundary fit
def fit_form(pairs):
    """pairs = [(L, n*)] with finite n*. Compare linear vs exponential by R^2."""
    pts = [(L, ns) for L, ns in pairs if ns and ns > 0]
    if len(pts) < 3:
        return {"verdict": "insufficient (need >=3 L with finite n*)", "points": pts}
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ybar = sum(ys) / len(ys); sst = sum((y - ybar) ** 2 for y in ys)
    def r2(pred): return 1 - sum((y - p) ** 2 for y, p in zip(ys, pred)) / sst if sst else 1.0
    N = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    b = (N * sxy - sx * sy) / (N * sxx - sx * sx); a = (sy - b * sx) / N
    r2_lin = r2([a + b * x for x in xs])
    ly = [math.log2(y) for y in ys]; sly = sum(ly); sxly = sum(x * v for x, v in zip(xs, ly))
    be = (N * sxly - sx * sly) / (N * sxx - sx * sx); lae = (sly - be * sx) / N
    r2_exp = r2([2 ** (lae + be * x) for x in xs]); base = 2 ** be
    if abs(b) < 0.5 and base < 1.4:
        v = "FLAT (n* independent of L)"
    elif r2_exp > r2_lin + 0.03 and base > 1.4:
        v = f"EXPONENTIAL  n*~c*{base:.2f}^L  (log-grade)"
    else:
        v = f"LINEAR  n*~{b:.1f}*L"
    return {"verdict": v, "linear_slope": round(b, 2), "linear_r2": round(r2_lin, 3),
            "exp_base_per_L": round(base, 2), "exp_r2": round(r2_exp, 3), "points": pts}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=5)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--kinds", nargs="+", default=["tf", "rnn", "ssm"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--out", default="three_arch_curriculum.json")
    args = ap.parse_args()

    gens = perm_gens(args.V, args.gens, DEVICE)
    print(f"device={DEVICE}  S_{args.V}  d={args.d}  chance={1/args.V:.3f}  "
          f"curriculum ns={args.ns}")
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}

    for kind in args.kinds:
        # RNN/SSM depth is not the associative-scan knob; still sweep for control,
        # but the KEY sweep is Transformer across L. RNN L is fixed at 1 by theory.
        Ls = args.Ls if kind == "tf" else [1]
        for L in Ls:
            reached_seeds = []
            for s in range(args.seeds):
                tag = f"{kind}_L{L}_s{s}"
                if tag in res:
                    reached_seeds.append(res[tag]["reached"]); continue
                t0 = time.time()
                hist, reached = curriculum_boundary(
                    kind, args.V, gens, args.d, L, args.ns, DEVICE,
                    steps=args.steps, seed=s, thresh=args.thresh)
                res[tag] = {"kind": kind, "L": L, "seed": s, "history": hist,
                            "reached": reached}
                json.dump(res, open(args.out, "w"), indent=2)
                reached_seeds.append(reached)
                print(f"  {tag}: n*={reached}  hist={ {k: round(v,2) for k,v in hist.items()} }"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            med = sorted(reached_seeds)[len(reached_seeds) // 2]
            res[f"{kind}_L{L}_median_nstar"] = med
            json.dump(res, open(args.out, "w"), indent=2)

    # -------- boundary fits per architecture --------
    print("\n=== boundary n*(L) fits ===")
    for kind in args.kinds:
        Ls = args.Ls if kind == "tf" else [1]
        pairs = [(L, res.get(f"{kind}_L{L}_median_nstar")) for L in Ls]
        pairs = [(L, ns) for L, ns in pairs if ns is not None]
        # a model that solves the whole ladder has n* = "beyond range" -> report
        top = max(args.ns)
        flat_high = all(ns >= top for _, ns in pairs) and pairs
        if flat_high:
            print(f"  {kind}: FLAT-HIGH (solved the whole ladder up to n={top} at "
                  f"every tested L) -> unbounded, true '!'")
            continue
        fit = fit_form(pairs)
        print(f"  {kind}: {fit['verdict']}    (n* per L: {pairs})")
    print("\nDecision: Transformer EXPONENTIAL + RNN FLAT-HIGH + SSM FLAT-LOW would")
    print("support 'the TC0 ceiling is shared but the route to it is set by each")
    print("architecture's forward category (adjunction placement)'.")


if __name__ == "__main__":
    main()
