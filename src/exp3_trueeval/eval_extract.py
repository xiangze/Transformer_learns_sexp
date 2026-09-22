"""
eval_extract.py -- pull M(u,v) and A(u,v) out of a live model.

Needs torch. Same code path on CPU and GPU (--device auto). Everything returned
is a plain numpy array, so eval_common.py never sees a tensor.

Design decisions that matter, and why:

1.  BASIS.  M is estimated in *value-token coordinates*, not raw d_model
    coordinates. Fix a set of k value tokens {t_1..t_k} spanning the semantic
    range of X = Y. The x-slot embedding is written
        e_x = sum_i c_i E[t_i],
    and the readout is the logit vector over the same k tokens. Then
        M = d(logits) / d c   in R^{k x k}
    has the SAME index set on both sides, so M(f) M(g) typechecks and E2 is
    well posed. A PCA basis would not compose.

2.  CONTINUITY.  c is a real vector, so the x-slot is continuously perturbable
    even though the underlying vocabulary is discrete. Without this the whole
    Jacobian construction degenerates -- discrete slots make an operator family
    indistinguishable from a table by construction.

3.  FIXED LAYOUT.  A(u,v) is a T x T object. Comparing A across u requires
    identical sequence length and identical slot positions; templates are padded
    to a common T and the probe refuses to run otherwise. This also kills the
    position-permutation gauge, under which A is only covariant (A -> P A P^T).

4.  HEAD AGGREGATION.  A is collected across all (layer, head) and stacked. Per
    head statistics are exactly what the degeneracy results (Wen et al. 2023;
    Meloux et al. 2025) show to be non-identifiable, so the probe never reports
    a single head.
"""

import numpy as np
import torch


# ------------------------------------------------------------------ adapters

class ModelAdapter:
    """Minimum surface a model must expose. Subclass for your own stack."""
    d_model: int
    n_layers: int
    n_heads: int
    device: torch.device
    dtype: torch.dtype

    def embed(self, ids):
        """(1,T) long -> (1,T,d) float"""
        raise NotImplementedError

    def value_embeddings(self, ids):
        """(k,) long -> (k,d) rows of the input embedding matrix"""
        raise NotImplementedError

    def forward_from_embeds(self, embeds, need_attn=False):
        """(1,T,d) -> (hidden (1,T,d) at readout layer, attentions or None)

        attentions: (n_layers, n_heads, T, T) tensor if need_attn."""
        raise NotImplementedError

    def readout(self, h):
        """(d,) hidden at the readout position -> (k,) logits over value tokens"""
        raise NotImplementedError


class HFAdapter(ModelAdapter):
    """transformers AutoModelForCausalLM. Works for natural-language pretrained
    checkpoints and for anything trained with the HF trainer."""

    def __init__(self, model, value_token_ids, readout_layer=-1,
                 device="auto", dtype=torch.float32):
        self.model = model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.to(self.device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = model.config
        self.d_model = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.readout_layer = readout_layer
        self.value_ids = torch.as_tensor(value_token_ids, device=self.device)
        self._W_U = self.model.get_output_embeddings().weight[self.value_ids]  # (k,d)
        self._E = self.model.get_input_embeddings().weight                     # (V,d)

    def embed(self, ids):
        return self._E[ids.to(self.device)].unsqueeze(0)

    def value_embeddings(self, ids=None):
        ids = self.value_ids if ids is None else torch.as_tensor(ids, device=self.device)
        return self._E[ids]

    def forward_from_embeds(self, embeds, need_attn=False):
        out = self.model(inputs_embeds=embeds,
                         output_hidden_states=True,
                         output_attentions=need_attn,
                         use_cache=False)
        h = out.hidden_states[self.readout_layer][0]          # (T,d)
        A = None
        if need_attn:
            A = torch.stack([a[0] for a in out.attentions])   # (L,H,T,T)
        return h, A

    def readout(self, h):
        return self._W_U @ h


class LocalAdapter(ModelAdapter):
    """For your own S-expression model. Expects an nn.Module with
        .tok_emb  (nn.Embedding)
        .lm_head  (nn.Linear, weight (V,d))
        .forward_from_embeds(x, need_attn) -> (hidden (1,T,d), attn (L,H,T,T)|None)
    Adapt the three attribute names below if yours differ."""

    def __init__(self, model, value_token_ids, device="auto", dtype=torch.float32):
        self.model = model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.model.to(self.device, dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._E = model.tok_emb.weight
        self.d_model = self._E.shape[1]
        self.value_ids = torch.as_tensor(value_token_ids, device=self.device)
        self._W_U = model.lm_head.weight[self.value_ids]
        self.n_layers = len(getattr(model, "blocks", []))
        self.n_heads = getattr(model, "n_heads", 1)

    def embed(self, ids):
        return self._E[ids.to(self.device)].unsqueeze(0)

    def value_embeddings(self, ids=None):
        ids = self.value_ids if ids is None else torch.as_tensor(ids, device=self.device)
        return self._E[ids]

    def forward_from_embeds(self, embeds, need_attn=False):
        h, A = self.model.forward_from_embeds(embeds, need_attn=need_attn)
        return h[0], A

    def readout(self, h):
        return self._W_U @ h


# ------------------------------------------------------------------ prompt slots

class Slots:
    """A fixed-layout prompt.

    ids       (T,) long, the full token sequence with placeholders already in place
    f_pos     list[int]  positions carrying the code u (may be several tokens)
    x_pos     int        the single position whose embedding is perturbed
    read_pos  int        position whose hidden state is read out (default -1)
    """

    def __init__(self, ids, f_pos, x_pos, read_pos=-1):
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.f_pos = list(f_pos)
        self.x_pos = int(x_pos)
        self.read_pos = int(read_pos) % len(self.ids)
        self.T = len(self.ids)


def check_layout(slot_list):
    """Refuse to compare A across prompts of different shape."""
    T = {s.T for s in slot_list}
    xp = {s.x_pos for s in slot_list}
    rp = {s.read_pos for s in slot_list}
    if len(T) != 1 or len(xp) != 1 or len(rp) != 1:
        raise ValueError(
            f"layout not fixed: T={T}, x_pos={xp}, read_pos={rp}. "
            "Pad all templates to a common length and keep slot positions "
            "identical, otherwise A(u) are not comparable as matrices.")


# ------------------------------------------------------------------ M(u,v)

@torch.no_grad()
def _forward_c(adapter, slot, c, V_emb):
    e = adapter.embed(slot.ids).clone()
    e[0, slot.x_pos] = (c[:, None] * V_emb).sum(0)
    h, _ = adapter.forward_from_embeds(e, need_attn=False)
    return adapter.readout(h[slot.read_pos])


def jacobian_M(adapter, slot, c0, V_emb, mode="jvp", eps=1e-3):
    """M = d(value-token logits) / d c  at c = c0, shape (k_out, k_in).

    mode="jvp"  forward-mode, k_in passes, exact.
    mode="fd"   central differences, 2*k_in passes. Use when the model has
                custom autograd that forward-mode does not support.

    Also returns the affine offset b = f(c0) - M c0, so that the local model is
    r ~ M c + b and E2 can compose affinely, as in the LRE parameterisation.
    """
    k = V_emb.shape[0]
    c0 = c0.to(V_emb.dtype)

    if mode == "jvp":
        def fn(c):
            e = adapter.embed(slot.ids).clone()
            e = e.to(V_emb.dtype)
            e[0, slot.x_pos] = (c[:, None] * V_emb).sum(0)
            h, _ = adapter.forward_from_embeds(e, need_attn=False)
            return adapter.readout(h[slot.read_pos])
        cols = []
        I = torch.eye(k, device=V_emb.device, dtype=V_emb.dtype)
        for i in range(k):
            _, jv = torch.func.jvp(fn, (c0,), (I[i],))
            cols.append(jv)
        M = torch.stack(cols, dim=1)                       # (k_out, k_in)
        r0 = fn(c0)
    else:
        cols = []
        for i in range(k):
            cp, cm = c0.clone(), c0.clone()
            cp[i] += eps
            cm[i] -= eps
            cols.append((_forward_c(adapter, slot, cp, V_emb)
                         - _forward_c(adapter, slot, cm, V_emb)) / (2 * eps))
        M = torch.stack(cols, dim=1)
        r0 = _forward_c(adapter, slot, c0, V_emb)

    b = r0 - M @ c0
    return M.detach().float().cpu().numpy(), b.detach().float().cpu().numpy()


# ------------------------------------------------------------------ A(u,v)

@torch.no_grad()
def attention_A(adapter, slot, c, V_emb, layers=None, heads=None, logit_coords=True):
    """Stacked attention maps at the same operating point as M.

    Returns (n_sel, T, T) numpy.

    logit_coords: A is row-stochastic, so the rows live on a simplex and a raw
    SVD of stacked A picks up an artefact rank drop from the sum constraint.
    The centred-log-ratio transform maps each row into the tangent space of the
    simplex first. Turn off only if you want the raw matrices.
    """
    e = adapter.embed(slot.ids).clone().to(V_emb.dtype)
    e[0, slot.x_pos] = (c[:, None] * V_emb).sum(0)
    _, A = adapter.forward_from_embeds(e, need_attn=True)
    if A is None:
        raise RuntimeError("adapter returned no attentions")
    A = A.float()                                          # (L,H,T,T)
    if layers is not None:
        A = A[layers]
    if heads is not None:
        A = A[:, heads]
    A = A.reshape(-1, A.shape[-2], A.shape[-1])
    if logit_coords:
        logA = torch.log(A.clamp_min(1e-9))
        A = logA - logA.mean(dim=-1, keepdim=True)
    return A.cpu().numpy()


# ------------------------------------------------------------------ grid sweep

def sweep_grid(adapter, slot_of, codes_u, codes_v, V_emb, c0_of,
               want_A=True, jac_mode="jvp", layers=None, heads=None,
               progress=None):
    """Walk the (u, v) grid once, collecting both sides.

    slot_of(u, v) -> Slots
    c0_of(v)      -> (k,) expansion point in value coordinates

    Returns dict of numpy arrays:
        M    (N_u, N_v, k, k)
        b    (N_u, N_v, k)
        A    (N_u, N_v, n_sel, T, T)   or None
    """
    slots = [slot_of(u, v) for u in codes_u for v in codes_v]
    check_layout(slots)

    Nu, Nv = len(codes_u), len(codes_v)
    Ms, bs, As = [], [], []
    for iu, u in enumerate(codes_u):
        rowM, rowb, rowA = [], [], []
        for iv, v in enumerate(codes_v):
            s = slot_of(u, v)
            c0 = c0_of(v).to(V_emb.device)
            M, b = jacobian_M(adapter, s, c0, V_emb, mode=jac_mode)
            rowM.append(M)
            rowb.append(b)
            if want_A:
                rowA.append(attention_A(adapter, s, c0, V_emb,
                                        layers=layers, heads=heads))
            if progress:
                progress(iu, iv)
        Ms.append(rowM)
        bs.append(rowb)
        if want_A:
            As.append(rowA)

    out = dict(M=np.asarray(Ms), b=np.asarray(bs))
    out["A"] = np.asarray(As) if want_A else None
    return out
