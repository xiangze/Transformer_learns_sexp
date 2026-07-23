"""
grade_v2.py
=====================================================================
Closing confound (B) and testing hypothesis (C).

(B) CYCLE SHORTCUT — closed two ways:
    1. HIDDEN CHAIN: the intermediate iterates pi^1(s)..pi^(r-1)(s) are NOT
       emitted into the context. The model sees only the k example pairs, the
       start symbol s, and a run of r "step" markers; it must emit pi^r(s) at the
       end. Nothing about the orbit is visible, so cycle detection on a visible
       prefix is impossible. Reuse must happen internally.
    2. LARGE V: with V=64 (vs 8), orbits rarely close within r<=8, so even
       partial visibility would not help.

(C) IS THE REUSE BUDGET IN SOFTMAX RATHER THAN DEPTH?
    If the non-natural Markov copy of softmax supplies the effective duplication
    budget, then r-dependence should appear in ATTENTION capacity knobs
    (temperature / number of heads) rather than in depth L.
    We sweep heads in {1,2,4,8} and an inverse-temperature scale in
    {0.5,1,2,4} at fixed depth, and compare the size of the r-interaction
    against the depth sweep.

DECISION RULE (stated before running)
    * If depth shows no r-interaction AND heads/temperature DO  -> (C), (A) refuted.
    * If nothing shows an r-interaction and accuracy is flat in r -> consistent
      with (A) (no graded structure anywhere) OR a task ceiling; report as such.
    * If depth shows an r-interaction once (B) is closed -> the earlier negative
      was a measurement artifact; graded reading revives.
"""

import math, json, argparse, time
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEP = 0          # reserved token id for the "apply once more" marker


def make_batch(B, V, k, r, device=DEVICE):
    """Hidden chain:  x1 y1 ... xk yk | s  STEP*r  -> answer pi^r(s).

    CRITICAL: the permutation is a CYCLE ON THE k SHOWN SYMBOLS, so every iterate
    pi^j(s) stays inside the shown pairs and the answer is always DETERMINED by
    the context. (Without this, chains escape the shown set and most targets are
    unanswerable, capping accuracy in a way that mimics a depth limit.)
    Intermediates are never emitted, so no cycle shortcut on a visible prefix.
    To stop the answer being read off a fixed cycle length, k is the cycle length
    and r < k, and symbol identities are re-randomised every sample."""
    T = 2 * k + 1 + r + 1
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        syms = (torch.randperm(V, device=device) + 1)[:k]      # k distinct symbols
        # pi = the k-cycle syms[0]->syms[1]->...->syms[k-1]->syms[0]
        nxt = {syms[i].item(): syms[(i + 1) % k].item() for i in range(k)}
        order = torch.randperm(k, device=device)               # shuffle pair order
        p = 0
        for i in order.tolist():
            x = syms[i].item()
            toks[b, p] = x; toks[b, p + 1] = nxt[x]; p += 2
        s = syms[torch.randint(0, k, (1,)).item()].item()
        toks[b, p] = s; p += 1
        for _ in range(r):
            toks[b, p] = STEP; p += 1
        cur = s
        for _ in range(r):
            cur = nxt[cur]
        toks[b, p] = cur
    return toks, T - 1


class Block(nn.Module):
    def __init__(self, d, h, mult=4, invtemp=1.0):
        super().__init__()
        self.h, self.dh, self.invtemp = h, d // h, invtemp
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
        s = (q @ k.transpose(-2, -1)) * (self.invtemp / math.sqrt(self.dh))
        s = s.masked_fill(~m, float("-inf"))
        o = (s.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, d)
        x = x + self.Wo(o)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, V, d=64, h=4, L=3, maxT=64, mult=4, invtemp=1.0):
        super().__init__()
        self.tok = nn.Embedding(V + 1, d); self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([Block(d, h, mult, invtemp) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V + 1, bias=False)

    def forward(self, t):
        x = self.tok(t) + self.pos(torch.arange(t.shape[1], device=t.device))[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))


def run(V, k, r, L, d, heads, steps, mult=4, invtemp=1.0, B=256, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    maxT = 2 * k + 2 + r + 2
    m = LM(V, d, heads, L, maxT=maxT, mult=mult, invtemp=invtemp).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    for _ in range(steps):
        toks, ap = make_batch(B, V, k, r)
        lg = m(toks)
        loss = F.cross_entropy(lg[:, ap - 1, :], toks[:, ap])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        toks, ap = make_batch(1024, V, k, r)
        lg = m(toks)
        acc = (lg[:, ap - 1, :].argmax(-1) == toks[:, ap]).float().mean().item()
    return acc, sum(p.numel() for p in m.parameters())
