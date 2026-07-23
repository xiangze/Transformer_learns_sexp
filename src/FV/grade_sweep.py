"""
grade_sweep.py
=====================================================================
Does a Transformer's capacity to REUSE an in-context value scale with DEPTH?

Hypothesis (graded exponential !_L)
    If the alternation  attention (linear/Markov) -> MLP (cartesian/PL) -> ...
    unfolds the LNL exponential once per layer, then depth L bounds the number of
    reuses r that a network can perform:  success iff  r <~ c*L.
    Prediction: the failure boundary in the (r, L) plane is a LINE through the
    origin, not a flat threshold in r, and not explained by parameter count.

Falsifier / control
    The obvious rival is "capacity is set by PARAMETER COUNT, not depth". So we
    run a WIDTH control: a shallow-but-wide model matched (or over-matched) in
    parameters to a deep-narrow one. If the wide-shallow model matches the deep
    one at high r, depth is not what matters and the graded reading is refuted.

Task: chained function application, which FORCES r sequential reuses
    A random bijection pi over V symbols is shown via k example pairs. Then a
    start symbol s is given and the model must emit the orbit
        pi(s), pi^2(s), ..., pi^r(s).
    Each step reuses the SAME inferred pi on the previous output. Unlike a
    multi-query prompt (where queries are independent lookups), this cannot be
    solved by r parallel retrievals: step j depends on step j-1. r = reuse depth.
    Scored on the LAST element pi^r(s) (all-or-nothing on the hardest step).
"""

import math, json, argparse, itertools
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(B, V, k, r, device=DEVICE):
    """x1 y1 ... xk yk | s  pi(s) pi^2(s) ... pi^r(s).  Returns toks, answer positions."""
    T = 2 * k + 1 + r
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        perm = torch.randperm(V, device=device)
        shown = torch.randperm(V, device=device)[:k]
        p = 0
        for i in range(k):
            toks[b, p] = shown[i]; toks[b, p + 1] = perm[shown[i]]; p += 2
        s = shown[torch.randint(0, k, (1,)).item()]
        toks[b, p] = s
        cur = s
        for j in range(r):
            cur = perm[cur]
            toks[b, p + 1 + j] = cur
    ans_pos = list(range(2 * k + 1, 2 * k + 1 + r))   # positions holding pi^1..pi^r
    return toks, ans_pos


class Block(nn.Module):
    def __init__(self, d, h, mult=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False); self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False); self.Wo = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, mult * d); self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x):
        B, T, d = x.shape
        z = self.ln1(x)
        q = self.Wq(z).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.Wk(z).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.Wv(z).view(B, T, self.h, self.dh).transpose(1, 2)
        m = torch.tril(torch.ones(T, T, device=x.device)).bool()
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s = s.masked_fill(~m, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, d)
        x = x + self.Wo(o)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, V, d=64, h=4, L=3, maxT=48, mult=4):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([Block(d, h, mult) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)

    def forward(self, t):
        x = self.tok(t) + self.pos(torch.arange(t.shape[1], device=t.device))[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_eval(V, k, r, L, d, heads, steps, B=192, lr=3e-3, mult=4, seed=0):
    torch.manual_seed(seed)
    maxT = 2 * k + 1 + r + 2
    m = LM(V, d, heads, L, maxT=maxT, mult=mult).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        toks, ans = make_batch(B, V, k, r)
        lg = m(toks)
        loss = sum(F.cross_entropy(lg[:, p - 1, :], toks[:, p]) for p in ans) / len(ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        toks, ans = make_batch(1024, V, k, r)
        lg = m(toks)
        last = ans[-1]
        acc = (lg[:, last - 1, :].argmax(-1) == toks[:, last]).float().mean().item()
    return acc, n_params(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--reuses", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--mode", default="sweep", choices=["sweep", "control"])
    args = ap.parse_args()

    if args.mode == "sweep":
        res = {}
        print(f"(r,L) sweep   V={args.V} k={args.k} d={args.d} steps={args.steps}"
              f"   chance={1/args.V:.3f}")
        print("     " + "".join(f"  L={L}  " for L in args.depths))
        for r in args.reuses:
            row = []
            for L in args.depths:
                acc, npar = train_eval(args.V, args.k, r, L, args.d, args.heads,
                                       args.steps)
                row.append(acc); res[f"r{r}_L{L}"] = {"acc": acc, "params": npar}
                print(f"r={r} L={L}: {acc:.3f}", flush=True)
            print(f"r={r} |" + "".join(f" {a:.3f} " for a in row), flush=True)
        json.dump(res, open("/home/claude/grade_sweep.json", "w"), indent=2)

    else:
        # WIDTH CONTROL: deep-narrow vs shallow-wide at matched/greater params
        print("width control (does depth or parameter count set reuse capacity?)")
        out = {}
        for r in args.reuses:
            deep, pdeep = train_eval(args.V, args.k, r, 4, 48, 4, args.steps)
            wide, pwide = train_eval(args.V, args.k, r, 1, 128, 8, args.steps, mult=8)
            out[f"r{r}"] = {"deep_L4_d48": deep, "deep_params": pdeep,
                            "wide_L1_d128": wide, "wide_params": pwide}
            print(f"r={r}: deep(L=4,d=48,{pdeep/1000:.0f}k)={deep:.3f}   "
                  f"wide(L=1,d=128,{pwide/1000:.0f}k)={wide:.3f}", flush=True)
        json.dump(out, open("/home/claude/grade_control.json", "w"), indent=2)


if __name__ == "__main__":
    main()
