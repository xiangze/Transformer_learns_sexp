"""
grade_gpu.py — large-scale (r, L, heads, temperature) sweep for GPU
=====================================================================
Question: where does the capacity for REUSING an in-context value live?
    (A) nowhere graded   (B) depth L   (C) attention capacity (heads/temperature)

CPU pilot findings this scales up (V=8, k=4, d=64, 800 steps, 1 seed):
    depth:  L=1 fails flat (~0.33) for r=1,2,3;  L=2 solves all of r=1,2,3
            -> depth is a FIXED threshold (L>=2), no r-interaction
    heads:  at L=2, h=1 gives r=1:1.000  r=2:1.000  r=3:0.234
            h=2 gives r=3:0.598 ;  h=8 gives r=3:0.999
            -> head count DOES interact with r  (supports (C))
    caveat: h=2,r=4 came out 1.000, i.e. non-monotone in r -> single-seed noise;
            THIS is the thing the GPU run must settle with multiple seeds.

Task (confounds closed)
    Hidden chain: show k pairs of a permutation that is a k-CYCLE ON THE SHOWN
    SYMBOLS (so every iterate stays inside the context and the target is always
    determined), then a start symbol and r STEP markers; the model must emit
    pi^r(s). Intermediates are never emitted -> no cycle shortcut on a visible
    prefix. Loss on the final token only -> supervision is constant in r.

Usage
    python grade_gpu.py --sweep depth   --seeds 5
    python grade_gpu.py --sweep heads   --seeds 5
    python grade_gpu.py --sweep temp    --seeds 5
    python grade_gpu.py --sweep all --rs 1 2 3 4 6 8 --Ls 1 2 4 8 --heads 1 2 4 8
Results append to a JSON checkpoint and the run is resumable.
"""

import math, json, os, time, argparse, itertools
import torch, torch.nn as nn, torch.nn.functional as F

STEP = 0


# ------------------------------------------------------------------ data
def make_batch(B, V, k, r, device):
    """Vectorised batch construction (the CPU pilot built samples in a Python
    loop; on GPU that dominates runtime, so this is fully tensorised)."""
    T = 2 * k + 1 + r + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    # k distinct symbols per row, from 1..V
    syms = torch.argsort(torch.rand(B, V, device=device), dim=1)[:, :k] + 1  # [B,k]
    nxt = torch.roll(syms, shifts=-1, dims=1)                                # cycle
    order = torch.argsort(torch.rand(B, k, device=device), dim=1)            # pair order
    xs = torch.gather(syms, 1, order)
    ys = torch.gather(nxt, 1, order)
    toks[:, 0:2 * k:2] = xs
    toks[:, 1:2 * k:2] = ys
    # start symbol: index j into the cycle
    j = torch.randint(0, k, (B,), device=device)
    s = torch.gather(syms, 1, j[:, None]).squeeze(1)
    toks[:, 2 * k] = s
    toks[:, 2 * k + 1:2 * k + 1 + r] = STEP
    ans = torch.gather(syms, 1, ((j + r) % k)[:, None]).squeeze(1)   # pi^r(s)
    toks[:, T - 1] = ans
    return toks, T - 1


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, h, mult=4, invtemp=1.0):
        super().__init__()
        self.h, self.dh, self.invtemp = h, d // h, invtemp
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, mult * d); self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x):
        B, T, d = x.shape
        z = self.ln1(x)
        q, k, v = self.Wqkv(z).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        # scale carries the inverse temperature (the (C) knob)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.invtemp / math.sqrt(self.dh))
        o = o.transpose(1, 2).contiguous().view(B, T, d)
        x = x + self.Wo(o)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, V, d, h, L, maxT, mult=4, invtemp=1.0):
        super().__init__()
        self.tok = nn.Embedding(V + 1, d); self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([Block(d, h, mult, invtemp) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V + 1, bias=False)

    def forward(self, t):
        x = self.tok(t) + self.pos(torch.arange(t.shape[1], device=t.device))[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))


# ------------------------------------------------------------------ train
def run(V, k, r, L, d, heads, steps, device, mult=4, invtemp=1.0,
        B=512, lr=3e-3, seed=0, amp=True, eval_n=8192):
    torch.manual_seed(seed)
    maxT = 2 * k + 2 + r + 2
    m = LM(V, d, heads, L, maxT, mult, invtemp).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp and device.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    m.train()
    for _ in range(steps):
        toks, ap = make_batch(B, V, k, r, device)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=(amp and device.type == "cuda")):
            lg = m(toks)
            loss = F.cross_entropy(lg[:, ap - 1, :].float(), toks[:, ap])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
    m.eval()
    correct = tot = 0
    with torch.no_grad():
        for _ in range(max(1, eval_n // 2048)):
            toks, ap = make_batch(2048, V, k, r, device)
            pred = m(toks)[:, ap - 1, :].argmax(-1)
            correct += (pred == toks[:, ap]).sum().item(); tot += 2048
    return correct / tot, sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ sweeps
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="all",
                    choices=["depth", "heads", "temp", "width", "all"])
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rs", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--heads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--temps", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default="grade_gpu_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  chance={1/args.V:.4f}")
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}

    def cell(tag, r, L, h, t):
        """Run all seeds for one configuration, with resume."""
        if tag in res: 
            print(f"  skip {tag}"); return
        accs = []
        t0 = time.time()
        for s in range(args.seeds):
            a, npar = run(args.V, args.k, r, L, args.d, h, args.steps, device,
                          invtemp=t, seed=s)
            accs.append(a)
        mean = sum(accs) / len(accs)
        var = sum((x - mean) ** 2 for x in accs) / len(accs)
        res[tag] = {"acc_mean": mean, "acc_std": var ** 0.5, "accs": accs,
                    "params": npar, "r": r, "L": L, "heads": h, "invtemp": t}
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  {tag}: {mean:.3f} ± {var**0.5:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    if args.sweep in ("depth", "all"):
        print("\n[depth sweep]  does required L grow with r?  (pilot said NO)")
        for r, L in itertools.product(args.rs, args.Ls):
            cell(f"depth_r{r}_L{L}", r, L, 4, 1.0)

    if args.sweep in ("heads", "all"):
        print("\n[head sweep]  does required head count grow with r?  (pilot said YES)")
        for r, h in itertools.product(args.rs, args.heads):
            cell(f"heads_r{r}_h{h}", r, 2, h, 1.0)

    if args.sweep in ("temp", "all"):
        print("\n[temperature sweep]  does softmax sharpness interact with r?")
        for r, t in itertools.product(args.rs, args.temps):
            cell(f"temp_r{r}_t{t}", r, 2, 4, t)

    if args.sweep in ("width", "all"):
        print("\n[width control]  is depth reducible to parameter count?")
        for r in args.rs:
            cell(f"width_deep_r{r}", r, 4, 4, 1.0)

    print(f"\nsaved -> {args.out}")
    print("Decision rule: an r-interaction in heads/temp but not depth supports (C);")
    print("no r-interaction anywhere supports (A); an r-interaction in depth revives grading.")


if __name__ == "__main__":
    main()
