"""
residual_reuse.py
=====================================================================
Does the residual connection carry the DEPTH-AXIS ADDITIVE COPY that the
categorical account (residual = biproduct diagonal Delta^oplus) predicts?

Theory under test
    residual: x |-> x + f(x)  uses the additive diagonal Delta^oplus: A -> A (+) A,
    i.e. it FANS OUT the running value across depth. Prediction: without a
    residual, an intermediate value cannot be re-read by many downstream
    consumers, so the model's ability to REUSE one inferred value r times should
    degrade, and degrade MORE as r grows (a residual x reuse-count interaction).
    A pointwise MLP cannot supply this: it is position-wise and cannot fan a
    value out across depth on its own.

What we explicitly do NOT test
    The runtime / memory footprint of programs an LLM *emits*. That is a property
    of generated text and has no theoretical link to the resource structure of the
    forward pass. Conflating the two would be a category error.

Confound control
    Ablating a residual at inference on a residual-trained model only measures
    distribution shift, so each architecture is TRAINED SEPARATELY (Exp 1).
    A same-weights probe is still possible, but only as a CONTINUOUS deformation
    of the residual coefficient alpha (Exp 2), which is a graded perturbation
    rather than an architecture the model never saw.

Task: in-context permutation with controlled reuse count r
    A random bijection pi is shown via k examples; then r queries must each be
    answered by APPLYING THE SAME pi. r is the number of times the inferred
    function must be reused. Reuse is the thing residuals should support.

Exp 1  train {full, none, attn-only, mlp-only} residual x r in {1,2,3}, separately.
Exp 2  alpha-sweep on the trained full-residual model (same weights).
"""

import math, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(B, V, k, r, device=DEVICE):
    """x1 y1 ... xk yk  q1 a1 ... qr ar ;  ai = pi(qi), queries drawn from shown."""
    T = 2 * (k + r)
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        perm = torch.randperm(V, device=device)
        shown = torch.randperm(V, device=device)[:k]
        p = 0
        for i in range(k):
            toks[b, p] = shown[i]; toks[b, p + 1] = perm[shown[i]]; p += 2
        qs = shown[torch.randint(0, k, (r,), device=device)]
        for q in qs:
            toks[b, p] = q; toks[b, p + 1] = perm[q]; p += 2
    return toks


class Block(nn.Module):
    def __init__(self, d, h, res_attn=True, res_mlp=True):
        super().__init__()
        self.h, self.dh = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False); self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False); self.Wo = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d); self.fc2 = nn.Linear(4 * d, d)
        self.res_attn, self.res_mlp = res_attn, res_mlp
        self.alpha = 1.0                      # residual coefficient (Exp 2)

    def attn(self, x):
        B, T, d = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s = s.masked_fill(~causal, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.Wo(o)

    def forward(self, x):
        a = self.attn(self.ln1(x))
        x = self.alpha * x + a if self.res_attn else a       # additive fan-out, or not
        m = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        x = self.alpha * x + m if self.res_mlp else m
        return x


class LM(nn.Module):
    def __init__(self, V, d=64, h=4, L=4, maxT=64, res_attn=True, res_mlp=True):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([Block(d, h, res_attn, res_mlp) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)

    def forward(self, t):
        x = self.tok(t) + self.pos(torch.arange(t.shape[1], device=t.device))[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

    def set_alpha(self, a):
        for b in self.blocks: b.alpha = a


def y_positions(k, r):
    """Indices of the r answer slots."""
    return [2 * (k + i) + 1 for i in range(r)]


def train(model, V, k, r, steps=2500, B=256, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ys = y_positions(k, r)
    model.train()
    for _ in range(steps):
        toks = make_batch(B, V, k, r)
        logits = model(toks)
        loss = sum(F.cross_entropy(logits[:, p - 1, :], toks[:, p]) for p in ys) / len(ys)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def accuracy(model, V, k, r, B=1024):
    model.eval()
    toks = make_batch(B, V, k, r)
    logits = model(toks)
    per_q = []
    for p in y_positions(k, r):
        per_q.append((logits[:, p - 1, :].argmax(-1) == toks[:, p]).float().mean().item())
    return per_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    configs = {
        "full":      dict(res_attn=True,  res_mlp=True),
        "none":      dict(res_attn=False, res_mlp=False),
        "attn_only": dict(res_attn=True,  res_mlp=False),
        "mlp_only":  dict(res_attn=False, res_mlp=True),
    }
    reuse_counts = [1, 2, 3]
    out = {"config": vars(args), "exp1": {}, "exp2": {}}

    print("=" * 66)
    print("EXP 1 — separately trained architectures  (mean final-query acc over seeds)")
    print(f"        chance = {1/args.V:.3f}")
    print("=" * 66)
    print(f"{'residual':>10} |" + "".join(f"   r={r}  " for r in reuse_counts))
    print("-" * 66)
    for name, cfg in configs.items():
        row, out["exp1"][name] = [], {}
        for r in reuse_counts:
            accs = []
            for s in range(args.seeds):
                torch.manual_seed(1000 * s + r)
                m = LM(args.V, args.d, 4, args.L, maxT=2 * (args.k + 4), **cfg).to(DEVICE)
                train(m, args.V, args.k, r, steps=args.steps)
                accs.append(accuracy(m, args.V, args.k, r)[-1])   # hardest: last query
            mean = sum(accs) / len(accs)
            row.append(mean)
            out["exp1"][name][f"r{r}"] = {"mean": mean, "seeds": accs}
        print(f"{name:>10} |" + "".join(f"  {a:.3f} " for a in row))

    # Exp 2 — same weights, continuous deformation of alpha on the full model
    print()
    print("=" * 66)
    print("EXP 2 — SAME WEIGHTS, residual coefficient alpha swept (full model, r=3)")
    print("        alpha=1 is the trained setting; alpha->0 removes the fan-out")
    print("=" * 66)
    torch.manual_seed(0)
    m = LM(args.V, args.d, 4, args.L, maxT=2 * (args.k + 4), **configs["full"]).to(DEVICE)
    train(m, args.V, args.k, 3, steps=args.steps)
    print(f"{'alpha':>7} | " + "  ".join(f"  q{i}  " for i in range(3)))
    print("-" * 40)
    for a in [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]:
        m.set_alpha(a)
        accs = accuracy(m, args.V, args.k, 3)
        out["exp2"][f"alpha_{a}"] = accs
        print(f"{a:>7.2f} | " + "  ".join(f"{x:.3f}" for x in accs))
    m.set_alpha(1.0)

    with open("/home/claude/residual_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> residual_results.json")


if __name__ == "__main__":
    main()
