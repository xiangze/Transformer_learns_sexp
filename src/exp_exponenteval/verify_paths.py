"""Direct verification: how much does v actually move the attention pattern
vs the MLP output?  This measures the v-path without ablation, so it is an
independent check on the freeze-based 2x2 decomposition.

Reported per layer, relative:
    sA  = ||dA/dv||   / ||A||          attention probabilities
    sM  = ||dmlp/dv|| / ||mlp||        MLP output
Averaged over programs and over 3 random v directions.
"""
import numpy as np, torch, sys
from model_num import ModelCfg, NumTransformer, PathCtl
from sexp_cont import *

def rel_sens(ckpt, n_u=24, n_dir=3, seed=0, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    m = NumTransformer(ModelCfg(**ck["cfg"])).to(device); m.load_state_dict(ck["state"]); m.eval()
    op = torch.from_numpy(out_positions()); npz = num_positions()
    rng = np.random.default_rng(seed)
    L = len(m.blocks)
    sA = np.zeros(L); sM = np.zeros(L); cnt = 0
    for _ in range(n_u):
        u = sample_program(rng)
        ids = torch.from_numpy(encode_with_v(u, [0,0,0])[0]).unsqueeze(0)
        v0 = rng.normal(size=3)
        for _ in range(n_dir):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            vf = torch.tensor(v0.astype(np.float32), requires_grad=True)
            vals = torch.zeros(1, ids.shape[1]).index_put(
                (torch.zeros(3,dtype=torch.long), torch.from_numpy(npz)), vf)
            _, cache = m(ids, vals, op, PathCtl(capture=False))
            # recompute with capture but keeping graph: do a manual forward
            x = m.emb(ids, vals); T = ids.shape[1]
            causal = m.causal[:,:,:T,:T]
            As, Ms = [], []
            import math, torch.nn.functional as F
            for blk in m.blocks:
                hx = blk.ln1(x)
                q,k,vv = blk._split(blk.wq(hx)), blk._split(blk.wk(hx)), blk._split(blk.wv(hx))
                att = (q @ k.transpose(-2,-1))/math.sqrt(blk.dh)
                att = att.masked_fill(causal, float("-inf"))
                probs = att.softmax(-1); As.append(probs)
                ao = (probs @ vv).transpose(1,2).reshape(1,T,-1)
                x = x + blk.wo(ao)
                mo = blk.fc_out(F.gelu(blk.fc_in(blk.ln2(x)))); Ms.append(mo)
                x = x + mo
            dt = torch.from_numpy(d.astype(np.float32))
            for l in range(L):
                gA = torch.autograd.grad((As[l]*torch.ones_like(As[l])).sum(), vf,
                                         retain_graph=True, allow_unused=True)[0]
                # directional derivative via JVP-style: use vector-Jacobian with random probe
                pA = torch.randn_like(As[l]); pM = torch.randn_like(Ms[l])
                gA = torch.autograd.grad((As[l]*pA).sum(), vf, retain_graph=True)[0]
                gM = torch.autograd.grad((Ms[l]*pM).sum(), vf, retain_graph=True)[0]
                sA[l] += float(abs(gA @ dt)) / (float(As[l].norm())*float(pA.norm())+1e-12)
                sM[l] += float(abs(gM @ dt)) / (float(Ms[l].norm())*float(pM.norm())+1e-12)
            cnt += 1
    return sA/cnt, sM/cnt

for ck in sys.argv[1:]:
    a, mm = rel_sens(ck)
    print(f"{ck}")
    for l in range(len(a)):
        print(f"   layer {l}:  rel |dA/dv| = {a[l]:.3e}   rel |dMLP/dv| = {mm[l]:.3e}"
              f"   ratio MLP/A = {mm[l]/(a[l]+1e-30):6.1f}x")
