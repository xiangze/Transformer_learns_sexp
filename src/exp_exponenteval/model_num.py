"""model_num.py -- small decoder-only Transformer with a continuous value channel,
plus the path-control machinery needed by the eval probes.

Two things distinguish this from a stock small Transformer:

1.  NumEmbed: <NUM> token positions get  e = tok_emb + phi(v),  so the forward
    pass is differentiable in the argument v.  phi has a linear part and an
    optional Fourier part.  The linear part alone is enough for linear
    programs and gives the cleanest Jacobians; the Fourier part is available
    for languages that need sharper value resolution.

2.  PathCtl: every block can have its MLP output and/or its attention
    probabilities *frozen* at values captured from a reference forward pass.
    A frozen tensor is detached, so v cannot flow through that path.  This is
    what implements the 2x2 path decomposition

        M_full      (nothing frozen)
        M_noMLP     (MLP outputs frozen)     -> v-path through MLP removed
        M_noA       (attn probs frozen)      -> v-path through the pattern removed
        M_val       (both frozen)            -> pure value path

    MLP outputs can also be mean-ablated, for the layerwise sweep.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from sexp_cont import N_DIM, NUM_ID, PAD_ID, SEQ_LEN, VOCAB_SIZE


# --------------------------------------------------------------------------
@dataclass
class PathCtl:
    """Per-layer overrides applied during a forward pass."""
    freeze_mlp_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    freeze_attn_probs: Dict[int, torch.Tensor] = field(default_factory=dict)
    freeze_values: Dict[int, torch.Tensor] = field(default_factory=dict)
    ablate_mlp: Set[int] = field(default_factory=set)      # replace MLP out by mean_mlp
    mean_mlp: Dict[int, torch.Tensor] = field(default_factory=dict)
    capture: bool = False


@dataclass
class ModelCfg:
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 512
    n_freq: int = 8
    numemb: str = "both"        # 'linear' | 'fourier' | 'both'
    dropout: float = 0.0


# --------------------------------------------------------------------------
class NumEmbed(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB_SIZE, cfg.d_model, padding_idx=PAD_ID)
        self.pos = nn.Parameter(torch.zeros(SEQ_LEN, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        if cfg.numemb in ("linear", "both"):
            self.lin = nn.Linear(1, cfg.d_model, bias=False)
        if cfg.numemb in ("fourier", "both"):
            self.register_buffer("omega", torch.logspace(-1.0, 0.5, cfg.n_freq))
            self.fproj = nn.Linear(2 * cfg.n_freq, cfg.d_model, bias=False)

    def forward(self, ids: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        e = self.tok(ids) + self.pos[: ids.shape[1]]
        phi = torch.zeros_like(e)
        if hasattr(self, "lin"):
            phi = phi + self.lin(vals.unsqueeze(-1))
        if hasattr(self, "fproj"):
            z = vals.unsqueeze(-1) * self.omega
            phi = phi + self.fproj(torch.cat([z.sin(), z.cos()], dim=-1))
        mask = (ids == NUM_ID).unsqueeze(-1).to(e.dtype)
        return e + phi * mask


class Block(nn.Module):
    def __init__(self, cfg: ModelCfg, layer: int):
        super().__init__()
        self.layer = layer
        self.cfg = cfg
        self.h = cfg.n_head
        self.dh = cfg.d_model // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.fc_in = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc_out = nn.Linear(cfg.d_ff, cfg.d_model)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x, causal, ctl: PathCtl, cache: dict):
        B, T, D = x.shape
        hx = self.ln1(x)
        q, k, v = self._split(self.wq(hx)), self._split(self.wk(hx)), self._split(self.wv(hx))
        if ctl.capture:
            cache.setdefault("values", {})[self.layer] = v.detach()
        if self.layer in ctl.freeze_values:
            v = ctl.freeze_values[self.layer]
            if v.shape[0] != B:
                v = v.expand(B, -1, -1, -1)
        if self.layer in ctl.freeze_attn_probs:
            probs = ctl.freeze_attn_probs[self.layer]
            if probs.shape[0] != B:
                probs = probs.expand(B, -1, -1, -1)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            att = att.masked_fill(causal, float("-inf"))
            probs = att.softmax(dim=-1)
        if ctl.capture:
            cache.setdefault("attn_probs", {})[self.layer] = probs.detach()
        ao = (probs @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.wo(ao)

        # ---- MLP (Geva-style key-value memory: fc_in rows = keys, fc_out cols = values)
        hx2 = self.ln2(x)
        act = F.gelu(self.fc_in(hx2))
        if ctl.capture:
            cache.setdefault("ffn_act", {})[self.layer] = act.detach()
        if self.layer in ctl.freeze_mlp_out:
            m = ctl.freeze_mlp_out[self.layer]
            if m.shape[0] != B:
                m = m.expand(B, -1, -1)
        elif self.layer in ctl.ablate_mlp:
            m = ctl.mean_mlp[self.layer].to(x.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(0).expand(B, -1, -1)
        else:
            m = self.fc_out(act)
        if ctl.capture:
            cache.setdefault("mlp_out", {})[self.layer] = m.detach()
        return x + m


class NumTransformer(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.emb = NumEmbed(cfg)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", mask.view(1, 1, SEQ_LEN, SEQ_LEN))

    def forward(self, ids, vals, out_pos, ctl: Optional[PathCtl] = None):
        ctl = ctl or PathCtl()
        cache: dict = {}
        x = self.emb(ids, vals)
        T = ids.shape[1]
        causal = self.causal[:, :, :T, :T]
        for blk in self.blocks:
            x = blk(x, causal, ctl, cache)
        x = self.ln_f(x)
        y = self.head(x).squeeze(-1)            # [B, T]
        y = y[:, out_pos]                       # [B, N_DIM]
        return y, cache


# --------------------------------------------------------------------------
# helpers for the probes
# --------------------------------------------------------------------------
@torch.no_grad()
def capture_reference(model, ids, vals, out_pos) -> dict:
    """Run once and capture attn probs / mlp outs to freeze later."""
    ctl = PathCtl(capture=True)
    _, cache = model(ids, vals, out_pos, ctl)
    return cache


#: Which of the three v-carrying paths each mode FREEZES at the reference.
#: Freezing a path detaches it, so v cannot flow through it.  The argument v
#: reaches the <OUT> positions only via attention (the <NUM> residual itself is
#: at a different position), so freezing all three must drive dF/dv to zero --
#: mode 'frozen' exists to check exactly that, and a nonzero Jacobian there
#: means the freezing is leaking.
#:
#:   one path removed  -> noA, noV, noMLP   (directly comparable: same count)
#:   one path left     -> mlpOnly, patOnly, val
PATH_FREEZE = {
    "full":    (),
    "noA":     ("attn_probs",),
    "noV":     ("values",),
    "noMLP":   ("mlp_out",),
    "val":     ("attn_probs", "mlp_out"),            # pure value path
    "patOnly": ("values", "mlp_out"),                # pure pattern path
    "mlpOnly": ("attn_probs", "values"),             # pure MLP path
    "frozen":  ("attn_probs", "values", "mlp_out"),  # validation: must give dF/dv = 0
}

_FIELD = {"attn_probs": "freeze_attn_probs",
          "values": "freeze_values",
          "mlp_out": "freeze_mlp_out"}


def ctl_for(mode: str, ref: dict, n_layer: int) -> PathCtl:
    """Build a PathCtl freezing the paths PATH_FREEZE assigns to `mode`."""
    if mode not in PATH_FREEZE:
        raise ValueError(f"unknown path mode {mode!r}; "
                         f"choose from {sorted(PATH_FREEZE)}")
    ctl = PathCtl()
    for key in PATH_FREEZE[mode]:
        setattr(ctl, _FIELD[key], {l: ref[key][l] for l in range(n_layer)})
    return ctl
