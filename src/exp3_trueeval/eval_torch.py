"""
eval_torch.py -- GPU path for the (l_in, l_out) scan.

Same probe protocol as eval_probe.TinyProbe, so eval_layers.scan2d /
position_transfer / fv_transfer_matrix work unchanged:

    probe.M(ids, x_pos, read_pos, l_in, l_out, ep, v=..., patch=..., return_spread=...)
    probe.head_out(ids, layer, pos)
    probe.n_layers, probe.k, probe.B

Why this is not just "the numpy code with .cuda()":

1.  BATCHED CENTRAL DIFFERENCES.  One cell of the scan needs 2k+1 forwards per
    jitter draw (k = size of the value-token basis). The numpy version runs them
    one at a time, which is the wrong shape for a GPU. Here all 2k+1
    perturbations and all n_jitter draws go in ONE batch: with k=16 and
    n_jitter=8 that is 264 rows of a single forward. The whole speedup is this,
    not the device.

2.  EARLY EXIT.  A cell with small l_out does not need the layers above it. A
    hook on block l_out raises a sentinel to stop the forward, so a full
    triangular scan costs roughly half of what running every cell to the top
    would.

3.  fp32 FOR THE JACOBIAN.  Central differences with eps ~ 1e-3 on fp16/bf16
    activations is below the noise floor -- bf16 has ~3 decimal digits. Weights
    may be loaded in bf16, but the probe forces fp32 autocast off. This is not
    optional; it is the difference between measuring M and measuring rounding.

Injection above layer 0 needs a frame in residual coordinates; build_in_basis
supplies one, averaged over prompts so it does not depend on u (a per-prompt
frame expresses each M(u) in its own coordinates and silently destroys r_op and
rho).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-12


class _StopForward(Exception):
    """Sentinel for early exit; carries the captured hidden state."""

    def __init__(self, h):
        self.h = h


# ------------------------------------------------------------------ model spec

@dataclass
class ModelSpec:
    """What the probe needs from a decoder-only LM."""
    forward_embeds: Callable            # (B,T,d) -> anything (output ignored)
    layers: nn.ModuleList               # blocks, for hook attachment
    embed_tokens: Callable              # (T,) long -> (T,d)
    final_norm: Callable                # (...,d) -> (...,d)
    unembed_rows: Callable              # (k,) long -> (k,d)
    n_layers: int
    d_model: int
    attn_of: Optional[Callable] = None  # layer index -> per-head outputs, for FV


def hf_spec(model, dtype=torch.float32):
    """ModelSpec for a transformers AutoModelForCausalLM.

    Covers the llama-style (model.model.layers + model.model.norm) and
    gpt2/gptj-style (transformer.h + transformer.ln_f) layouts; anything else
    raises rather than guessing, because a wrong `layers` list would make the
    layer indices in the scan meaningless.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers, norm = model.model.layers, model.model.norm
        embed = model.model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers, norm = model.transformer.h, model.transformer.ln_f
        embed = model.transformer.wte
    else:
        raise ValueError(
            "unrecognised layout: expected model.model.layers or "
            "model.transformer.h. Pass a ModelSpec explicitly.")
    W_U = model.get_output_embeddings().weight

    return ModelSpec(
        forward_embeds=lambda e: model(inputs_embeds=e, use_cache=False),
        layers=layers,
        embed_tokens=lambda ids: embed(ids),
        final_norm=lambda h: norm(h),
        unembed_rows=lambda vid: W_U[vid],
        n_layers=len(layers),
        d_model=model.config.hidden_size,
    )


# ------------------------------------------------------------------ tiny torch LM

class _Block(nn.Module):
    """Pre-RMSNorm attention + MLP, no per-channel gain (folded into the next
    matrix, which is what makes the residual-rotation gauge exact)."""

    def __init__(self, d, H, f):
        super().__init__()
        self.H, self.dh = H, d // H
        self.Wq, self.Wk, self.Wv = (nn.Parameter(torch.empty(H, d, d // H))
                                     for _ in range(3))
        self.Wo = nn.Parameter(torch.empty(d, d))
        self.W1, self.W2 = nn.Parameter(torch.empty(d, f)), nn.Parameter(torch.empty(f, d))

    @staticmethod
    def rms(x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def attn_heads(self, h):
        hn = self.rms(h)
        T = h.shape[-2]
        mask = torch.triu(torch.full((T, T), -1e9, device=h.device,
                                     dtype=h.dtype), 1)
        outs = []
        for hd in range(self.H):
            q, k, v = hn @ self.Wq[hd], hn @ self.Wk[hd], hn @ self.Wv[hd]
            a = torch.softmax(q @ k.transpose(-1, -2) / self.dh ** 0.5 + mask, -1)
            outs.append(a @ v)
        return outs

    def forward(self, h):
        h = h + torch.cat(self.attn_heads(h), -1) @ self.Wo
        return h + nn.functional.gelu(self.rms(h) @ self.W1, approximate="tanh") @ self.W2


class TinyTorchLM(nn.Module):
    """Torch port of eval_probe.TinyLM, used to cross-validate the GPU path
    against the numpy reference. Not a research model."""

    def __init__(self, V=64, d=32, L=4, H=4, f=64, T=24, seed=0):
        super().__init__()
        self.V, self.d, self.T = V, d, T
        g = torch.Generator().manual_seed(seed)
        self.E = nn.Parameter(torch.empty(V, d))
        self.P = nn.Parameter(torch.empty(T, d))
        self.layers = nn.ModuleList([_Block(d, H, f) for _ in range(L)])
        self.WU = nn.Parameter(torch.empty(d, V))
        self._init(g)

    def _init(self, g):
        def s(*sh):
            return torch.randn(*sh, generator=g) / sh[-2] ** 0.5
        with torch.no_grad():
            self.E.copy_(s(self.V, self.d))
            self.P.copy_(s(self.T, self.d) * 0.1)
            for blk in self.layers:
                for name in ("Wq", "Wk", "Wv"):
                    getattr(blk, name).copy_(s(blk.H, self.d, blk.dh))
                blk.Wo.copy_(s(self.d, self.d))
                blk.W1.copy_(s(self.d, blk.W1.shape[-1]))
                blk.W2.copy_(s(blk.W1.shape[-1], self.d))
            self.WU.copy_(s(self.d, self.V))

    def forward(self, inputs_embeds=None, use_cache=False):
        h = inputs_embeds
        for blk in self.layers:
            h = blk(h)
        return h

    def spec(self):
        return ModelSpec(
            forward_embeds=lambda e: self(inputs_embeds=e),
            layers=self.layers,
            embed_tokens=lambda ids: self.E[ids] + self.P[:len(ids)],
            final_norm=_Block.rms,
            unembed_rows=lambda vid: self.WU.T[vid],
            n_layers=len(self.layers),
            d_model=self.d,
            attn_of=lambda li, h: self._head_out(li, h),
        )

    def _head_out(self, li, h):
        blk = self.layers[li]
        outs = blk.attn_heads(h)
        res = []
        for hd, o in enumerate(outs):
            z = torch.zeros_like(h)
            z[..., hd * blk.dh:(hd + 1) * blk.dh] = o
            res.append(z @ blk.Wo)
        return torch.stack(res)                 # (H, T, d)


# ------------------------------------------------------------------ probe

class TorchProbe:
    """GPU/CPU probe. device="cuda" and device="cpu" run the identical code."""

    def __init__(self, spec, value_ids, B_in, x_pos=None, device="auto",
                 dtype=torch.float32, eps=1e-3, batch_size=256):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.dtype = dtype
        self.spec = spec
        self.eps = float(eps)
        self.batch_size = int(batch_size)
        self.value_ids = torch.as_tensor(value_ids, dtype=torch.long,
                                         device=self.dev)
        self.k = len(value_ids)
        self.n_layers = spec.n_layers
        self.B = {l: torch.as_tensor(np.asarray(v), dtype=dtype, device=self.dev)
                  for l, v in B_in.items()}
        with torch.no_grad():
            self.R = spec.unembed_rows(self.value_ids).to(dtype).detach()

    # -- batched forward with injection, early exit ---------------------

    @torch.no_grad()
    def _run(self, ids, l_in, l_out, inject, patch=None, x_pos=None,
             read_pos=None):
        """inject: (B, d) vectors written into position x_pos at layer l_in.
        Returns (B, k) value-token logits read at l_out."""
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.dev)
        e = self.spec.embed_tokens(ids_t).to(self.dtype)         # (T,d)
        Bn = inject.shape[0]
        e = e.unsqueeze(0).expand(Bn, -1, -1).clone()

        if l_in == 0:
            e[:, x_pos, :] = inject
        if patch is not None and patch[0] == 0:
            e[:, patch[1], :] = e[:, patch[1], :] + patch[2]

        handles, captured = [], {}

        def pre_hook(_m, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            h = h.clone()
            h[:, x_pos, :] = inject
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h
            return args, kwargs

        def patch_hook(_m, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            h = h.clone()
            h[:, patch[1], :] = h[:, patch[1], :] + patch[2]
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h
            return args, kwargs

        def stop_hook(_m, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            raise _StopForward(h)

        if l_in > 0:
            handles.append(self.spec.layers[l_in].register_forward_pre_hook(
                pre_hook, with_kwargs=True))
        if patch is not None and patch[0] > 0:
            handles.append(self.spec.layers[patch[0]].register_forward_pre_hook(
                patch_hook, with_kwargs=True))
        if l_out < self.n_layers:
            handles.append(self.spec.layers[l_out].register_forward_pre_hook(
                stop_hook, with_kwargs=True))

        try:
            out = self.spec.forward_embeds(e)
            h = out if torch.is_tensor(out) else out.hidden_states[-1]
        except _StopForward as s:
            h = s.h
        finally:
            for hd in handles:
                hd.remove()

        hr = self.spec.final_norm(h[:, read_pos, :].to(self.dtype))
        return hr @ self.R.T                                      # (B,k)

    # -- M ---------------------------------------------------------------

    def M(self, ids, x_pos, read_pos, l_in, l_out, ep=None, v=0, patch=None,
          return_spread=False):
        if l_out <= l_in:
            raise ValueError("l_out must be > l_in: with no block between "
                             "injection and readout, M is identically zero")
        B = self.B[l_in]
        k = self.k
        cs = ep.draws(v) if ep is not None else [np.eye(k)[v]]
        c0 = torch.as_tensor(np.stack(cs), dtype=self.dtype, device=self.dev)
        nd = c0.shape[0]

        # rows: nd * (2k + 1) -- plus/minus per coordinate, then the base point
        I = torch.eye(k, dtype=self.dtype, device=self.dev)
        plus = c0[:, None, :] + self.eps * I[None]                # (nd,k,k)
        minus = c0[:, None, :] - self.eps * I[None]
        grid = torch.cat([plus.reshape(-1, k), minus.reshape(-1, k), c0], 0)

        outs = []
        for s in range(0, grid.shape[0], self.batch_size):
            chunk = grid[s:s + self.batch_size]
            outs.append(self._run(ids, l_in, l_out, chunk @ B, patch=patch,
                                  x_pos=x_pos, read_pos=read_pos))
        y = torch.cat(outs, 0)                                    # (nd*(2k+1), k)

        n = nd * k
        yp = y[:n].reshape(nd, k, k)
        ym = y[n:2 * n].reshape(nd, k, k)
        yb = y[2 * n:]
        Ms = ((yp - ym) / (2 * self.eps)).transpose(1, 2)         # (nd,k_out,k_in)
        bs = yb - torch.einsum("nij,nj->ni", Ms, c0)

        M = Ms.mean(0).double().cpu().numpy()
        b = bs.mean(0).double().cpu().numpy()
        if return_spread:
            spread = (0.0 if nd == 1 else float(
                (Ms - Ms.mean(0, keepdim=True)).flatten(1).norm(dim=1).std()
                / (Ms.mean(0).norm() + EPS)))
            return M, b, spread
        return M, b

    # -- FV extraction ---------------------------------------------------

    @torch.no_grad()
    def head_out(self, ids, layer, pos):
        if self.spec.attn_of is None:
            raise NotImplementedError(
                "this ModelSpec exposes no per-head outputs; for an HF model "
                "wire attn_of to the o_proj input split by head")
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=self.dev)
        e = self.spec.embed_tokens(ids_t).to(self.dtype).unsqueeze(0)
        h = e
        for li in range(layer):
            h = self.spec.layers[li](h)
        ho = self.spec.attn_of(layer, h)                          # (H,1,T,d)
        return ho[:, 0, pos % len(ids), :].double().cpu().numpy()


# ------------------------------------------------------------------ frames

@torch.no_grad()
def build_in_basis(probe_spec, prompts, layer, x_pos, value_ids, device,
                   dtype=torch.float32):
    """Fixed frame at `layer`, one direction per value token, averaged over
    prompts so the frame does not depend on u."""
    k = len(value_ids)
    acc = torch.zeros(k, probe_spec.d_model, dtype=dtype, device=device)
    for i, tok in enumerate(value_ids):
        for ids in prompts:
            ids = np.asarray(ids).copy()
            ids[x_pos] = tok
            ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
            h = probe_spec.embed_tokens(ids_t).to(dtype).unsqueeze(0)
            for li in range(layer):
                h = probe_spec.layers[li](h)
            acc[i] += h[0, x_pos]
        acc[i] /= len(prompts)
    return acc.double().cpu().numpy()
