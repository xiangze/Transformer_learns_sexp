"""
retrieval.py — Which component carries in-context retrieval? (Experiment I)
=====================================================================
Goal (from the discussion): the categorical eval-apply structure — softmax as a
DYNAMIC Markov kernel that ROUTES, and the internal-hom EVAL that applies — is
meant to capture RETRIEVAL, the regime where Transformers have a comparative
advantage over RNN/SSM (which compress context into a fixed-size state and so
cannot address an arbitrary earlier item). This experiment does NOT merely check
that retrieval is solved (that is known); it ISOLATES which part does it, and
whether Transformers beat fixed-state models specifically here.

Task: in-context function LOOKUP + apply  (Function-Vector-like, retrieval-heavy)
    The context defines m independent key->function bindings:
        k_1 : f_1 ,  k_2 : f_2 , ... , k_m : f_m
    where each f_i is a random bijection on a value alphabet, presented as a few
    (input, output) demonstration pairs. Then a QUERY  (k_j, x)  asks for f_j(x).
    Solving requires (a) ROUTING to the right binding k_j among m distractors
    (dynamic sparsity: attend to 1 of m), then (b) APPLYING that function to x.
    m controls retrieval difficulty (number of distractors); this is the axis a
    fixed-size recurrent state should struggle with as m grows.

Two things measured
    1. THREE-ARCH: Transformer vs GRU-RNN vs diagonal-SSM as m grows. Prediction:
       Transformer stays high; fixed-state models degrade with m (can't address
       an arbitrary binding from a compressed state).  -> retrieval is where the
       Transformer advantage lives.
    2. COMPONENT ABLATION (Transformer only): which piece carries retrieval?
       - freeze attention to uniform  (kills dynamic routing / the Markov kernel)
       - linearize MLP (identity activation)  (kills the eval-realization Phi)
       - mean-ablate MLP  (kills value transform)
       Prediction from the eval-apply reading: freezing attention destroys
       retrieval (routing is the Markov-kernel dynamic sparsity); linearizing MLP
       hurts the APPLY value but not the routing.  A double dissociation would
       localize routing in softmax and apply-value in MLP.
"""

import math, json, argparse, time
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def make_batch(B, m, V, kd, device):
    """Layout (one vocab): [k_i, (a, f_i(a)) x kd] for i=1..m, then query (k_j, x)
    -> f_j(x). Keys are ids 0..m-1; values ids m..m+V-1. Query x is drawn from
    the demonstrated inputs of binding j, so the answer is determined by context.
    m = number of distractor bindings (retrieval difficulty / dynamic sparsity)."""
    return _rebuild(B, m, V, kd, device)


def _rebuild(B, m, V, kd, device):
    per = 1 + 2 * kd
    T = m * per + 2 + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    valbase = m
    ans = torch.zeros(B, dtype=torch.long, device=device)
    anspos = T - 1
    for b in range(B):
        p = 0
        chosen_demo = {}
        perms = [torch.randperm(V, device=device) for _ in range(m)]
        for i in range(m):
            toks[b, p] = i; p += 1
            demos = torch.randperm(V, device=device)[:kd]
            chosen_demo[i] = demos
            for a in demos:
                toks[b, p] = valbase + a
                toks[b, p + 1] = valbase + perms[i][a]
                p += 2
        j = torch.randint(0, m, (1,), device=device).item()
        # pick query x from the demonstrated inputs of binding j -> answerable
        x = chosen_demo[j][torch.randint(0, kd, (1,), device=device).item()]
        toks[b, p] = j; toks[b, p + 1] = valbase + x
        ans[b] = valbase + perms[j][x]
    toks[:, anspos] = ans
    return toks, anspos


# ------------------------------------------------------------------ Transformer with ablation hooks
class TFBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d = h, d
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.f1 = nn.Linear(d, 4 * d); self.f2 = nn.Linear(4 * d, d)
        self.freeze_attn = False; self.linear_mlp = False

    def forward(self, x):
        B, T, d = x.shape
        z = self.ln1(x)
        q, k, v = self.qkv(z).chunk(3, -1)
        dh = d // self.h
        q, k, v = [y.view(B, T, self.h, dh).transpose(1, 2) for y in (q, k, v)]
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        if self.freeze_attn:                       # kill dynamic routing (Markov kernel)
            att = causal.float(); att = att / att.sum(-1, keepdim=True)
            o = att[None, None] @ v
        else:
            s = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
            s = s.masked_fill(~causal, float("-inf"))
            o = s.softmax(-1) @ v
        x = x + self.o(o.transpose(1, 2).reshape(B, T, d))
        hmid = self.f1(self.ln2(x))
        hmid = hmid if self.linear_mlp else F.gelu(hmid)   # kill eval-realization Phi
        return x + self.f2(hmid)


class TF(nn.Module):
    def __init__(self, ntok, d, h, L, maxT):
        super().__init__()
        self.tok = nn.Embedding(ntok, d); self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([TFBlock(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, ntok)

    def forward(self, t):
        x = self.tok(t) + self.pos(torch.arange(t.shape[1], device=t.device))[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

    def set_ablation(self, freeze_attn=False, linear_mlp=False):
        for b in self.blocks:
            b.freeze_attn = freeze_attn; b.linear_mlp = linear_mlp


class RNN(nn.Module):
    def __init__(self, ntok, d, L):
        super().__init__()
        self.tok = nn.Embedding(ntok, d)
        self.rnn = nn.GRU(d, d, num_layers=L, batch_first=True)
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
    return {"tf": TF(ntok, d, h, L, maxT), "rnn": RNN(ntok, d, L),
            "ssm": DiagSSM(ntok, d, L)}[kind]


def run(kind, m, V, kd, L, d, steps, device, B=256, lr=3e-3, seed=0, h=4,
        ablation=None):
    torch.manual_seed(seed)
    ntok = m + V
    per = 1 + 2 * kd; maxT = m * per + 4
    net = build(kind, ntok, d, L, maxT, h).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr); net.train()
    for _ in range(steps):
        toks, ap = make_batch(B, m, V, kd, device)
        loss = F.cross_entropy(net(toks)[:, ap - 1, :], toks[:, ap])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    if ablation and kind == "tf":
        net.set_ablation(**ablation)
    with torch.no_grad():
        toks, ap = make_batch(2048, m, V, kd, device)
        acc = (net(toks)[:, ap - 1, :].argmax(-1) == toks[:, ap]).float().mean().item()
    if kind == "tf": net.set_ablation()
    return acc, sum(p.numel() for p in net.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=6)
    ap.add_argument("--kd", type=int, default=3)          # demos per binding
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mode", default="both", choices=["arch", "ablation", "both"])
    ap.add_argument("--out", default="retrieval_results.json")
    args = ap.parse_args()
    import os
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    print(f"device={DEVICE}  V={args.V} kd={args.kd}  chance={1/args.V:.3f}")

    if args.mode in ("arch", "both"):
        print("\n[three-arch: does the retrieval advantage need dynamic routing?]")
        for kind in ["tf", "rnn", "ssm"]:
            for m in args.ms:
                tag = f"arch_{kind}_m{m}"
                if tag in res: continue
                accs = [run(kind, m, args.V, args.kd, args.L, args.d, args.steps,
                            DEVICE, seed=s)[0] for s in range(args.seeds)]
                res[tag] = {"acc": sum(accs)/len(accs), "accs": accs}
                json.dump(res, open(args.out, "w"), indent=2)
                print(f"  {tag}: {res[tag]['acc']:.3f}", flush=True)

    if args.mode in ("ablation", "both"):
        print("\n[component ablation (Transformer): what carries retrieval?]")
        abls = {"intact": {}, "freeze_attn": {"freeze_attn": True},
                "linear_mlp": {"linear_mlp": True}}
        for m in args.ms:
            for name, ab in abls.items():
                tag = f"abl_{name}_m{m}"
                if tag in res: continue
                accs = [run("tf", m, args.V, args.kd, args.L, args.d, args.steps,
                            DEVICE, seed=s, ablation=ab)[0] for s in range(args.seeds)]
                res[tag] = {"acc": sum(accs)/len(accs), "accs": accs}
                json.dump(res, open(args.out, "w"), indent=2)
                print(f"  {tag}: {res[tag]['acc']:.3f}", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
