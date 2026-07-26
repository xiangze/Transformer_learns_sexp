"""
three_arch.py — Transformer vs simple RNN vs diagonal linear SSM
=====================================================================
Experiment II: does the failure boundary on state tracking distinguish the
architectures by their categorical forward structure?

Task: EXPLICIT permutation composition (unlike the earlier hidden-chain task,
    here the sequence of permutations to compose is fully given, so there is no
    in-context inference confound — this isolates SEQUENTIAL COMPOSITION, the
    NC1 core).
    Input: a start state x0 in {0..V-1}, then a sequence of n permutations
    delta_1..delta_n each drawn from a fixed generating set G of S_V; the model
    must output (delta_n o ... o delta_1)(x0).  n = composition length.

Predictions (from Merrill et al. TC0/NC1 + the categorical reading):
    * simple (non-linear) RNN  : solves all n with 1 layer      (true !)
    * Transformer, depth L     : solves n up to ~ c * 2^L        (associative
                                   scan = log-grade !_{log n})
    * diagonal linear SSM      : no better than Transformer, likely worse
                                   (no genuine state update; TC0)
Discriminator: FIT the failure boundary n*(L). Linear n* ~ cL  vs  exponential
    n* ~ c*2^L  vs  n*-independent-of-L. The FUNCTION FORM is the evidence.

This CPU pilot uses tiny V, n, L to check the three models are correctly wired
and that the discriminator has any resolving power. GPU version scales V,n,L,seeds.
"""

import math, json, argparse, time
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def perm_gens(V, n_gens, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(V, generator=g) for _ in range(n_gens)]).to(device)


def make_batch(B, V, n, gens, device):
    """x0 , g_{i1}, g_{i2}, ..., g_{in}  -> (g_in o ... o g_i1)(x0).
    Tokens: states 0..V-1, generators V..V+G-1. Answer at the last position."""
    G = gens.shape[0]
    T = 1 + n + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    x0 = torch.randint(0, V, (B,), device=device)
    toks[:, 0] = x0
    idx = torch.randint(0, G, (B, n), device=device)       # which generator each step
    toks[:, 1:1 + n] = idx + V
    state = x0.clone()
    for j in range(n):
        gj = gens[idx[:, j]]                               # [B,V]
        state = torch.gather(gj, 1, state[:, None]).squeeze(1)
    toks[:, T - 1] = state                                  # target state
    return toks, T - 1


# ------------------------------------------------------------------ models
class TF(nn.Module):
    def __init__(self, ntok, d, h, L, maxT):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(maxT, d)
        enc = lambda: nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), ln2=nn.LayerNorm(d),
            qkv=nn.Linear(d, 3 * d, bias=False), o=nn.Linear(d, d, bias=False),
            f1=nn.Linear(d, 4 * d), f2=nn.Linear(4 * d, d)))
        self.blocks = nn.ModuleList([enc() for _ in range(L)]); self.h = h; self.d = d
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, ntok)

    def forward(self, t):
        B, T = t.shape
        x = self.tok(t) + self.pos(torch.arange(T, device=t.device))[None]
        for b in self.blocks:
            z = b["ln1"](x); q, k, v = b["qkv"](z).chunk(3, -1)
            dh = self.d // self.h
            q, k, v = [y.view(B, T, self.h, dh).transpose(1, 2) for y in (q, k, v)]
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + b["o"](o.transpose(1, 2).reshape(B, T, self.d))
            x = x + b["f2"](F.gelu(b["f1"](b["ln2"](x))))
        return self.head(self.lnf(x))


class RNN(nn.Module):
    """Simple non-linear (GRU) RNN — expected to reach NC1 (true state update)."""
    def __init__(self, ntok, d, L):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        self.rnn = nn.GRU(d, d, num_layers=L, batch_first=True)
        self.head = nn.Linear(d, ntok)

    def forward(self, t):
        y, _ = self.rnn(self.tok(t))
        return self.head(y)


class DiagSSM(nn.Module):
    """Diagonal linear SSM (input-independent diagonal transition) — TC0, expected
    to be no better than Transformer at ordered composition."""
    def __init__(self, ntok, d, L):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        # diagonal decay a (per channel), input map B, output C
        self.a = nn.ParameterList([nn.Parameter(torch.rand(d) * 0.5 + 0.2) for _ in range(L)])
        self.Bp = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.Cp = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.head = nn.Linear(d, ntok)

    def forward(self, t):
        x = self.tok(t)
        B, T, d = x.shape
        for a, Bp, Cp, ln in zip(self.a, self.Bp, self.Cp, self.ln):
            u = Bp(ln(x))
            a_ = a.clamp(0.01, 0.999)
            h = torch.zeros(B, d, device=x.device); outs = []
            for j in range(T):
                h = a_ * h + u[:, j]          # diagonal, input-independent transition
                outs.append(h)
            y = Cp(torch.stack(outs, 1))
            x = x + y
        return self.head(x)


def build(kind, ntok, d, L, maxT, heads=4):
    if kind == "tf":   return TF(ntok, d, heads, L, maxT)
    if kind == "rnn":  return RNN(ntok, d, L)
    if kind == "ssm":  return DiagSSM(ntok, d, L)


def run(kind, V, n, L, d, gens, steps, device, B=256, lr=3e-3, seed=0, heads=4):
    torch.manual_seed(seed)
    ntok = V + gens.shape[0]; maxT = n + 4
    m = build(kind, ntok, d, L, maxT, heads).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train()
    for _ in range(steps):
        toks, ap = make_batch(B, V, n, gens, device)
        loss = F.cross_entropy(m(toks)[:, ap - 1, :], toks[:, ap])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        toks, ap = make_batch(2048, V, n, gens, device)
        acc = (m(toks)[:, ap - 1, :].argmax(-1) == toks[:, ap]).float().mean().item()
    return acc, sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=5)          # S_5 core (NC1-complete)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--kinds", nargs="+", default=["tf", "rnn", "ssm"])
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="three_arch.json")
    args = ap.parse_args()

    gens = perm_gens(args.V, args.gens, DEVICE, seed=123)
    print(f"device={DEVICE}  V={args.V} (S_{args.V}) gens={args.gens}  chance={1/args.V:.3f}")
    import os, json
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for kind in args.kinds:
        for L in args.Ls:
            for n in args.ns:
                tag = f"{kind}_L{L}_n{n}"
                if tag in res: continue
                accs = []
                t0 = time.time()
                for s in range(args.seeds):
                    a, npar = run(kind, args.V, n, L, args.d, gens, args.steps, DEVICE, seed=s)
                    accs.append(a)
                mean = sum(accs) / len(accs)
                res[tag] = {"acc": mean, "accs": accs, "params": npar,
                            "kind": kind, "L": L, "n": n}
                json.dump(res, open(args.out, "w"), indent=2)
                print(f"  {tag}: {mean:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
