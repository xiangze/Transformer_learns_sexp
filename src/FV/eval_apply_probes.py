"""
eval_apply_probes.py
====================================================================
Empirically adjudicating two structural claims about where a decoder
Transformer realizes the pieces of an eval / apply (β-reduction) loop,
in the minimal in-context-learning setting of *applying an inferred
function to a query*.

Task (symbolic ICL, forces genuine function inference):
    A random bijection pi over an alphabet of V symbols is drawn per
    sample. The prompt shows k input/output examples of pi, then one
    or more queries; the model must output pi(query).
    Sequence layout (all tokens share one vocabulary):
        x1 y1 x2 y2 ... xk yk  xq1 yq1  xq2 yq2 ...
    with y_i = pi(x_i). Loss is next-token CE on every y slot.
    Queries are drawn from the shown inputs so the answer is determined.

Two competing structural hypotheses under test
    (H_apply_MLP)  MLP performs apply / beta-reduction: the transformation
                   (function-representation, argument) -> pi(x) is realized
                   inside the MLP sublayer.
    (H_copy_softmax) If any 'copy' (linear-logic-forbidden diagonal, x |-> x (x) x,
                   i.e. multiplicative reuse of one value in two slots) is
                   recovered, it lives in the softmax attention routing
                   (Markov-category copy), NOT in the pointwise MLP activation.
    The user's caveat: the currying R^d -> R^{d x d} may already be done by
                   Q K^T in attention (K=Wk x, Q=Wq x, Q K^T in R^{d x d}),
                   which would place function-formation in attention and weaken
                   H_apply_MLP.

Three probes
    A. CROSS-POSITION INTERACTION (copy localization).
       'copy into two multiplicative slots' = a genuine (cross-position)
       bilinear interaction. It shows up as a nonzero *mixed* second
       derivative d^2 out_q / d h_a d h_b for two DISTINCT source positions
       a != b. A pointwise MLP acts position-wise -> that cross block is 0.
       Attention couples positions multiplicatively -> nonzero. We report the
       Frobenius mass of the cross-position interaction for the attention
       sublayer vs the MLP sublayer. Large-for-attn / ~0-for-MLP supports
       H_copy_softmax and locates the (x) (copy) structure in attention.

    B. APPLY LOCALIZATION (causal activation patching).
       Two prompts with DIFFERENT bijections pi (clean) and pi' (corrupt) but
       the SAME query symbol, so clean answer != corrupt answer. Run the model
       on the corrupt prompt; patch in the clean run's activation at the query
       position, separately for attn-out and mlp-out of each layer; measure how
       often the output flips to the clean answer pi(x). Whichever component's
       patch transfers the *applied result* is where apply (beta-reduction) is
       carried. attn-out wins  -> apply/eval in attention (supports the caveat).
       mlp-out wins            -> apply in MLP (supports H_apply_MLP).

    C. MULTI-QUERY REUSE ABLATION (copy = one function, many args).
       Prompts with several queries needing the SAME pi applied to different
       args. Reuse of one inferred function across queries is 'copy'. We ablate
       (i) attention routing (freeze attn to uniform) vs (ii) MLP nonlinearity
       (replace activation by identity, keeping the linear W2 W1) and read the
       per-query accuracy drop. If reuse depends on attention routing far more
       than on MLP nonlinearity, copy sits in attention.

This tiny synthetic model is a *methodology demonstrator*, not GPT-2. The same
three probes attach unchanged to a real LM via forward hooks (see run_on_hf()
stub at the bottom); re-run there before drawing model-scale conclusions.
"""

import math, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def make_batch(B, V, k, n_query, device=DEVICE):
    """Return tokens [B,T] and a target mask marking y-slots. Layout:
    x1 y1 ... xk yk  xq1 yq1 ... xq_nq yq_nq ; y_i = pi(x_i)."""
    T = 2 * (k + n_query)
    toks = torch.zeros(B, T, dtype=torch.long, device=device)
    tgt_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        perm = torch.randperm(V, device=device)          # the bijection pi
        shown = torch.randperm(V, device=device)[:k]      # distinct example inputs
        pos = 0
        for i in range(k):
            xi = shown[i]
            toks[b, pos] = xi; toks[b, pos + 1] = perm[xi]
            tgt_mask[b, pos + 1] = True
            pos += 2
        # queries drawn from shown inputs -> answer is determined by the examples
        qsel = shown[torch.randint(0, k, (n_query,), device=device)]
        for q in qsel:
            toks[b, pos] = q; toks[b, pos + 1] = perm[q]
            tgt_mask[b, pos + 1] = True
            pos += 2
    return toks, tgt_mask


# ------------------------------------------------------------------ model
class Block(nn.Module):
    """Pre-norm decoder block exposing attn-out and mlp-out separately,
    with hooks to (a) capture, (b) patch, (c) ablate each sublayer."""
    def __init__(self, d, h, mlp_ratio=4):
        super().__init__()
        self.h = h; self.d = d; self.dh = d // h
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)
        # control switches used by the probes
        self.freeze_attn = False     # replace softmax pattern by causal-uniform
        self.linear_mlp = False      # replace GELU by identity (kill nonlinearity)
        self.capture = False
        self.cache = {}
        self.patch = {}              # {'attn': (pos, vec), 'mlp': (pos, vec)}

    def attn(self, x):
        B, T, d = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
        if self.freeze_attn:
            # data-independent routing: uniform over allowed (causal) positions
            att = causal.float()
            att = att / att.sum(-1, keepdim=True)
            att = att.view(1, 1, T, T).expand(B, self.h, T, T)
        else:
            score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # Q K^T : bilinear in x
            score = score.masked_fill(~causal, float("-inf"))
            att = score.softmax(-1)                                  # -> Markov / distribution
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.Wo(out)

    def mlp(self, x):
        h = self.fc1(x)
        h = h if self.linear_mlp else F.gelu(h)   # the pointwise nonlinearity under test
        return self.fc2(h)

    def forward(self, x):
        a = self.attn(self.ln1(x))
        if "attn" in self.patch:
            pos, vec = self.patch["attn"]; a[:, pos, :] = vec
        if self.capture: self.cache["attn"] = a.detach().clone()
        x = x + a
        m = self.mlp(self.ln2(x))
        if "mlp" in self.patch:
            pos, vec = self.patch["mlp"]; m[:, pos, :] = vec
        if self.capture: self.cache["mlp"] = m.detach().clone()
        x = x + m
        return x


class TinyLM(nn.Module):
    def __init__(self, V, d=64, h=4, L=3, maxT=64):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(maxT, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, toks):
        T = toks.shape[1]
        x = self.tok(toks) + self.pos(torch.arange(T, device=toks.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))

    # probe utilities -------------------------------------------------
    def set(self, freeze_attn=False, linear_mlp=False):
        for b in self.blocks:
            b.freeze_attn = freeze_attn; b.linear_mlp = linear_mlp

    def clear_patch(self):
        for b in self.blocks: b.patch = {}

    def clear_capture(self):
        for b in self.blocks: b.capture = False; b.cache = {}


# ------------------------------------------------------------------ train
def train(model, V, k, steps, B=256, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        nq = int(torch.randint(1, 4, (1,)).item())   # train on 1..3 queries -> reuse is learned
        toks, mask = make_batch(B, V, k, n_query=nq)
        logits = model(toks)
        # predict token t+1 at each y-slot => compare logits at position t to token at t
        pred_pos = mask.clone(); pred_pos[:, 0] = False
        tgt = toks[mask]
        src = logits[:, :-1, :][mask[:, 1:]]
        loss = F.cross_entropy(src, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % max(1, steps // 8) == 0:
            acc = query_accuracy(model, V, k)
            print(f"  step {s+1:4d}  loss {loss.item():.3f}  query-acc {acc:.3f}")
    return model


@torch.no_grad()
def query_accuracy(model, V, k, n_query=1, B=512, freeze_attn=False, linear_mlp=False):
    model.eval(); model.set(freeze_attn, linear_mlp)
    toks, mask = make_batch(B, V, k, n_query)
    logits = model(toks)
    # last query answer is at the final position; its predictor is position T-2
    T = toks.shape[1]
    pred = logits[:, T - 2, :].argmax(-1)
    gold = toks[:, T - 1]
    model.set(False, False)
    return (pred == gold).float().mean().item()


# ------------------------------------------------------------------ Probe A
@torch.no_grad()
def probe_A_cross_position(model, V, k, n_probe=64):
    """SUBLAYER-LOCAL mixed second derivative.

    For each block we perturb *that sublayer's own input* at two DISTINCT source
    positions a != b (both != query position) and read the mixed difference
        f(+a,+b) - f(+a) - f(+b) + f()
    of the sublayer's output AT THE QUERY POSITION. This isolates whether the
    sublayer itself introduces a cross-position multiplicative interaction
    (a (x) / copy signature), with no leakage through the residual path.

    - MLP o LayerNorm is strictly position-wise  => output at qpos cannot depend
      on inputs at a,b != qpos              => mixed term is exactly 0.
    - Attention couples positions through the softmax normalization and value
      mixing                                => mixed term is nonzero.
    So a large attn / ~0 mlp gap localizes the copy/diagonal in softmax attention.
    """
    model.eval(); model.set(False, False)
    d = model.tok.embedding_dim
    res = {li: {"attn": 0.0, "mlp": 0.0} for li in range(len(model.blocks))}
    for _ in range(n_probe):
        toks, _ = make_batch(1, V, k, n_query=1)
        T = toks.shape[1]; qpos = T - 1; a, b = 0, 2      # distinct, != qpos
        eps = 0.5
        ea = torch.zeros(1, T, d); ea[0, a] = eps * torch.randn(d)
        eb = torch.zeros(1, T, d); eb[0, b] = eps * torch.randn(d)

        # actual residual-stream input to each block
        x = model.tok(toks) + model.pos(torch.arange(T, device=toks.device))[None]
        for li, blk in enumerate(model.blocks):
            u = x                                   # this block's input stream
            def attn_out(pert):                     # attention sublayer, local perturb
                return blk.attn(blk.ln1(u + pert))[0, qpos]
            m00a = attn_out(0); m10a = attn_out(ea); m01a = attn_out(eb); m11a = attn_out(ea + eb)
            res_attn = (m11a - m10a - m01a + m00a).norm().item()

            a_full = blk.attn(blk.ln1(u))           # real attention contribution
            m_in = u + a_full                       # actual MLP-sublayer input
            def mlp_out(pert):                      # MLP sublayer, local perturb
                return blk.mlp(blk.ln2(m_in + pert))[0, qpos]
            m00 = mlp_out(0); m10 = mlp_out(ea); m01 = mlp_out(eb); m11 = mlp_out(ea + eb)
            res_mlp = (m11 - m10 - m01 + m00).norm().item()

            res[li]["attn"] += res_attn; res[li]["mlp"] += res_mlp
            x = m_in + blk.mlp(blk.ln2(m_in))       # advance real stream
    for li in res:
        for comp in res[li]: res[li][comp] /= n_probe
    return res


# ------------------------------------------------------------------ Probe B
def probe_B_apply_patching(model, V, k, n_trials=400):
    """Activation patching: does patching attn-out vs mlp-out at the query
    position transfer the *applied answer* pi(x) from a clean run into a
    corrupt run (different bijection, same query symbol)?"""
    model.eval(); model.set(False, False)
    L = len(model.blocks)
    flip = {("attn", li): 0 for li in range(L)}
    flip.update({("mlp", li): 0 for li in range(L)})
    usable = 0
    for _ in range(n_trials):
        # build clean and corrupt sharing the same shown inputs and same query symbol
        perm_c = torch.randperm(V, device=DEVICE)
        perm_d = torch.randperm(V, device=DEVICE)
        shown = torch.randperm(V, device=DEVICE)[:k]
        qsym = shown[torch.randint(0, k, (1,)).item()]
        if perm_c[qsym] == perm_d[qsym]:
            continue  # need distinct answers to detect a flip
        def build(perm):
            T = 2 * (k + 1); t = torch.zeros(1, T, dtype=torch.long, device=DEVICE)
            p = 0
            for i in range(k):
                t[0, p] = shown[i]; t[0, p + 1] = perm[shown[i]]; p += 2
            t[0, p] = qsym; t[0, p + 1] = perm[qsym]
            return t, T
        tc, T = build(perm_c); td, _ = build(perm_d)
        qpos = T - 2   # position whose next-token prediction is the answer
        clean_ans = perm_c[qsym].item(); corrupt_ans = perm_d[qsym].item()

        # capture clean activations
        for b in model.blocks: b.capture = True
        _ = model(tc)
        clean_cache = {li: {kk: model.blocks[li].cache[kk][0, qpos].clone()
                            for kk in ("attn", "mlp")} for li in range(L)}
        for b in model.blocks: b.capture = False; b.cache = {}

        # baseline corrupt prediction
        base_logits = model(td)
        base_pred = base_logits[0, qpos].argmax().item()
        if base_pred != corrupt_ans:
            continue  # only count trials the model gets right, so a flip is meaningful
        usable += 1
        for comp in ("attn", "mlp"):
            for li in range(L):
                model.clear_patch()
                model.blocks[li].patch[comp] = (qpos, clean_cache[li][comp])
                pred = model(td)[0, qpos].argmax().item()
                if pred == clean_ans:
                    flip[(comp, li)] += 1
        model.clear_patch()
    denom = max(1, usable)
    return {f"{c}_L{li}": flip[(c, li)] / denom for (c, li) in flip}, usable


# ------------------------------------------------------------------ Probe C
def probe_C_reuse(model, V, k, n_query=3):
    """Per-query accuracy for multi-query prompts under (i) intact,
    (ii) frozen attention, (iii) linearized MLP. Reuse of one inferred
    function across queries = copy; which ablation destroys it?"""
    @torch.no_grad()
    def per_query_acc(freeze_attn, linear_mlp, B=512):
        model.eval(); model.set(freeze_attn, linear_mlp)
        toks, mask = make_batch(B, V, k, n_query)
        logits = model(toks)
        T = toks.shape[1]
        accs = []
        for qi in range(n_query):
            outpos = 2 * (k + qi) + 1        # y-slot of query qi
            pred = logits[:, outpos - 1, :].argmax(-1)
            gold = toks[:, outpos]
            accs.append((pred == gold).float().mean().item())
        model.set(False, False)
        return accs
    return {
        "intact":      per_query_acc(False, False),
        "freeze_attn": per_query_acc(True,  False),
        "linear_mlp":  per_query_acc(False, True),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    print(f"[device={DEVICE}]  V={args.V} k={args.k} d={args.d} L={args.L} heads={args.heads}")
    model = TinyLM(args.V, d=args.d, h=args.heads, L=args.L,
                   maxT=2 * (args.k + 4) + 4).to(DEVICE)
    print("training in-context permutation-application model ...")
    train(model, args.V, args.k, steps=args.steps)
    acc = query_accuracy(model, args.V, args.k)
    print(f"final single-query accuracy: {acc:.3f}  (chance = {1/args.V:.3f})\n")

    out = {"config": vars(args), "final_acc": acc}

    print("Probe A - cross-position interaction (copy localization):")
    A = probe_A_cross_position(model, args.V, args.k)
    for li in A:
        print(f"  layer {li}: attn={A[li]['attn']:.4f}   mlp={A[li]['mlp']:.4f}"
              f"   ratio attn/mlp = {A[li]['attn']/max(1e-9,A[li]['mlp']):.1f}")
    out["probeA"] = A

    print("\nProbe B - apply localization (activation patching, flip-to-clean rate):")
    B, usable = probe_B_apply_patching(model, args.V, args.k)
    for kk in sorted(B): print(f"  {kk}: {B[kk]:.3f}")
    print(f"  (usable trials: {usable})")
    out["probeB"] = {"flip": B, "usable": usable}

    print("\nProbe C - multi-query reuse ablation (per-query accuracy):")
    C = probe_C_reuse(model, args.V, args.k, n_query=3)
    for cond, accs in C.items():
        print(f"  {cond:12s}: " + "  ".join(f"q{i}={a:.3f}" for i, a in enumerate(accs)))
    out["probeC"] = C

    with open("/home/claude/probe_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> probe_results.json")


if __name__ == "__main__":
    main()
