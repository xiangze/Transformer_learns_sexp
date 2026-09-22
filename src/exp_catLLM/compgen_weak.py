"""
compgen_weak.py — weak compositional generalization probe (道B)
=====================================================================
Question (a NECESSARY condition for eval, not sufficient): does the model apply
a function to an argument compositionally, i.e. generalize to (f, x) PAIRS it
never saw together, when each f and each x WAS seen (in other combinations)?

This is the WEAK split by design (see caveats):
  * every function f_i and every argument x is seen during training,
  * but a held-out set of (i, x) COMBINATIONS is removed from training and
    tested. Success on held-out combinations is evidence the model composes
    "which function" with "which argument" rather than memorizing whole (f,x,y)
    triples.

Task (in-context function application, Function-Vector-like):
    context defines m bindings  k_i : f_i   (f_i a random bijection on V values,
    shown as ALL V demonstration pairs so f_i is fully specified in-context),
    then a query (k_j, x) -> f_j(x).
    Because every f_i is given in-context by its full table, a held-out (j,x)
    is still answerable BY APPLICATION; a pure memorizer that keyed on the
    (query-key, query-x) surface pair without using the in-context table would
    fail the held-out combinations.

Controls:
  * chance = 1/V.
  * "seen" test set (combinations present in training dist) vs "held-out" test
    set (combinations never trained). Gap = failure of compositionality.
  * we HOLD OUT combinations, not functions or arguments (that would be the
    STRONG split); this is explicitly the weak, preliminary version.

CPU-scale; single seed first, then a couple of seeds for a rough error bar.
"""

import math, json, os, time, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_functions(m, V, seed):
    g = torch.Generator().manual_seed(seed)
    # m fixed bijections on V symbols (values 0..V-1)
    return [torch.randperm(V, generator=g) for _ in range(m)]


def make_batch(B, m, V, perms, held, device, split="train"):
    """One vocab: keys 0..m-1, values m..m+V-1.
    Layout: for each i in 0..m-1:  key_i, then V demo pairs (a, f_i(a)) in random
    order; then query (key_j, x); answer f_j(x) at the last position.
    split='train'  -> sample (j,x) NOT in held
    split='heldout'-> sample (j,x) IN held
    split='seen'   -> sample (j,x) NOT in held (same dist as train; generalization control)
    """
    per = 1 + 2 * V
    T = m * per + 2 + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    valbase = m
    ans = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        p = 0
        for i in range(m):
            toks[b, p] = i; p += 1
            order = torch.randperm(V)
            for a in order.tolist():
                toks[b, p] = valbase + a
                toks[b, p + 1] = valbase + perms[i][a].item()
                p += 2
        # choose (j,x) per split
        while True:
            j = random.randrange(m); x = random.randrange(V)
            inheld = (j, x) in held
            if split == "heldout" and inheld: break
            if split in ("train", "seen") and not inheld: break
        toks[b, p] = j; toks[b, p + 1] = valbase + x
        ans[b] = valbase + perms[j][x].item()
    return toks, T - 1, ans


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


@torch.no_grad()
def evaluate(m_net, m, V, perms, held, device, split, B=1024):
    m_net.eval()
    toks, ap, ans = make_batch(B, m, V, perms, held, device, split=split)
    pred = m_net(toks)[:, ap - 1, :].argmax(-1)
    return (pred == ans).float().mean().item()


def run(m, V, d, L, heads, steps, held_frac, seed, device, B=128, lr=3e-3):
    torch.manual_seed(seed); random.seed(seed)
    perms = build_functions(m, V, seed)
    # held-out set of (j,x) combinations
    all_combos = [(j, x) for j in range(m) for x in range(V)]
    random.shuffle(all_combos)
    n_held = max(1, int(held_frac * len(all_combos)))
    held = set(all_combos[:n_held])
    ntok = m + V; per = 1 + 2 * V; maxT = m * per + 4
    net = TF(ntok, d, heads, L, maxT).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()
    for step in range(steps):
        toks, ap, ans = make_batch(B, m, V, perms, held, device, split="train")
        loss = F.cross_entropy(net(toks)[:, ap - 1, :], ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    seen = evaluate(net, m, V, perms, held, device, "seen")
    heldout = evaluate(net, m, V, perms, held, device, "heldout")
    return seen, heldout, len(held), len(all_combos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--V", type=int, default=6)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--held_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="compgen_weak.json")
    a = ap.parse_args()
    t0 = time.time()
    seen, held, nh, nt = run(a.m, a.V, a.d, a.L, a.heads, a.steps,
                             a.held_frac, a.seed, DEVICE)
    res = {}
    if os.path.exists(a.out): res = json.load(open(a.out))
    key = f"m{a.m}_V{a.V}_L{a.L}_h{a.heads}_hf{a.held_frac}_s{a.seed}"
    res[key] = {"seen_acc": round(seen, 3), "heldout_acc": round(held, 3),
                "chance": round(1 / a.V, 3), "n_held": nh, "n_total": nt,
                "gap": round(seen - held, 3)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"{key}: seen={seen:.3f} heldout={held:.3f} chance={1/a.V:.3f} "
          f"gap={seen-held:+.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
