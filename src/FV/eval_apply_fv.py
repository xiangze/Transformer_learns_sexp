"""
eval_apply_fv.py  --  Eval / Apply structure tests built on ericwtodd/function_vectors
=====================================================================================
Drop this into the `src/` directory of https://github.com/ericwtodd/function_vectors
(next to compute_indirect_effect.py) and run inside their `fv` conda env with a real
HF model (the paper uses gpt2-xl and gpt-j-6b). It reuses their FV extraction, their
`add_function_vector` intervention, their `baukit.TraceDict` hooks and their eval
metrics -- it does not reimplement any of that.

It operationalizes, in FV-native terms, the three structural claims we have been
testing:

  Typing under test
    * FV = a POINT v_t in P_FV subset R^d  (their compute_function_vector returns
      sum_{(L,H) in top heads} out_proj(head_act) in R^{resid_dim} -- literally a
      point in R^d, NOT an element of the internal hom R^{d x d}).
    * apply / beta-reduction = realizing that point into a function via some Phi and
      evaluating it on the query argument.
    * copy (linear-logic-forbidden diagonal / reuse of one value in many slots) --
      is it in softmax attention (Markov-category copy) or in the MLP nonlinearity?

  Test 1  REUSE + REALIZATION (copy + eval), via an edit-layer sweep.
      ONE fixed FV v_t is inserted (their add_function_vector) into MANY distinct
      zero-shot queries x_1..x_N. A single point yielding the correct per-argument
      answer f_t(x_i) across varying x_i IS reuse-across-arguments (copy) + eval.
      Sweeping the insertion layer separates "the point still needs downstream
      nonlinear realization Phi" (mid-layer peak) from "pure additive readout /
      bias only" (inserting at the last layer). Mid-peak >> last-layer  =>  the
      point is genuinely realized+applied downstream, not a constant logit bias.

  Test 2  APPLY LOCALIZATION, via component ablation downstream of insertion.
      With v_t inserted at the peak layer L*, ablate attention-out vs MLP-out at
      the query's last token for each layer l > L* and measure destruction of the
      FV-induced correct answer. MLP-downstream ablation destroys it  => MLP
      realizes Phi / apply (consistent with Geva et al. 2021 key-value memories).
      Attention-downstream destroys it  => eval/routing carried by attention
      (consistent with the Q K^T currying hypothesis).

  Test 3  COPY LOCALIZATION, via cross-position mixed second difference.
      Perturb the input embedding at two DISTINCT source token positions a != b
      and read the mixed difference  f(+a,+b)-f(+a)-f(+b)+f()  of each sublayer's
      output at the query position. A genuine x (x) x (copy into two multiplicative
      slots) shows up as a nonzero cross-position mixed term; a strictly
      position-wise MLP generates none of its own. See the CAVEAT in that function:
      in a real residual network the MLP INHERITS cross-position structure through
      the residual stream, so compare against the toy model (eval_apply_probes.py)
      for the exact position-wise isolation where MLP's own term is machine-zero.

None of these tests can be run without downloading a HF model, so this file is
written against the repo API and left for you to run on GCP. Every imported symbol
below was checked against the current repo source.
"""

import argparse
import numpy as np
import torch
from baukit import TraceDict

# ---- repo imports (run from within function_vectors/src) --------------------
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import (load_dataset, word_pairs_to_prompt_data,
                                create_prompt)
from utils.extract_utils import (get_mean_head_activations,
                                 compute_function_vector)
from utils.eval_utils import get_answer_id, compute_individual_token_rank
from utils.intervention_utils import add_function_vector
from compute_indirect_effect import compute_indirect_effect


# ---------------------------------------------------------------- architecture
def get_mlp_hook_names(model_config):
    """Per-architecture MLP-sublayer module names (the repo only ships
    attn_hook_names and layer_hook_names)."""
    name = model_config['name_or_path'].lower()
    L = model_config['n_layers']
    if 'gpt2' in name:
        return [f'transformer.h.{l}.mlp' for l in range(L)]
    if 'gpt-j' in name:
        return [f'transformer.h.{l}.mlp' for l in range(L)]
    if 'gpt-neox' in name or 'pythia' in name:
        return [f'gpt_neox.layers.{l}.mlp' for l in range(L)]
    if any(s in name for s in ('llama', 'gemma', 'olmo')):
        return [f'model.layers.{l}.mlp' for l in range(L)]
    raise ValueError(f"add MLP hook names for {name}")


def final_readout(model, model_config):
    """(ln_f, W_U) for the bias-only null in Test 1, per architecture."""
    name = model_config['name_or_path'].lower()
    if 'gpt2' in name or 'gpt-j' in name:
        return model.transformer.ln_f, model.lm_head
    if 'gpt-neox' in name or 'pythia' in name:
        return model.gpt_neox.final_layer_norm, model.embed_out
    return model.model.norm, model.lm_head


# ---------------------------------------------------------------- FV extraction
def extract_fv(dataset, model, model_config, tokenizer,
               n_top_heads=10, n_icl=10, n_trials=25, mean_trials=100):
    """Reproduce the repo's FV pipeline: mean head activations -> AIE ->
    compute_function_vector. Returns (fv in R^{1 x d}, top_heads)."""
    mean_acts = get_mean_head_activations(dataset, model, model_config, tokenizer,
                                          n_icl_examples=n_icl, N_TRIALS=mean_trials)
    aie, _ = compute_indirect_effect(dataset, mean_acts, model, model_config,
                                     tokenizer, n_shots=n_icl, n_trials=n_trials)
    fv, top_heads = compute_function_vector(mean_acts, aie, model, model_config,
                                            n_top_heads=n_top_heads)
    return fv, top_heads


# ---------------------------------------------------------------- prompt helper
def zeroshot_prompt(query_pair, model_config):
    """A single zero-shot prompt (query only, no ICL examples), repo-style."""
    pd = word_pairs_to_prompt_data({'input': [], 'output': []},
                                   query_target_pair=query_pair,
                                   prepend_bos_token=model_config['prepend_bos'])
    sentence = create_prompt(pd)
    target = pd['query_target']['output']
    target = target[0] if isinstance(target, list) else target
    return sentence, target


@torch.no_grad()
def clean_and_fv_logits(sentence, fv, edit_layer, model, model_config, tokenizer):
    """Last-token logits for the clean run and the FV-intervened run (their
    add_function_vector hook, edited on layer_hook_names)."""
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    clean = model(**inputs).logits[:, -1, :]
    hook = add_function_vector(edit_layer, fv.reshape(1, model_config['resid_dim']),
                               model.device, idx=-1)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=hook):
        interv = model(**inputs).logits[:, -1, :]
    return clean, interv


# ================================================================ Test 1
@torch.no_grad()
def test1_reuse_and_realization(dataset, fv, model, model_config, tokenizer,
                                n_eval=100, layers=None):
    """Insert ONE fixed FV into many distinct zero-shot queries; sweep edit layer.
    Also evaluate a bias-only null (project FV through the final readout and add
    the resulting CONSTANT logit vector) to show a query-independent shift cannot
    reproduce per-argument correctness."""
    if layers is None:
        layers = list(range(0, model_config['n_layers'], 2))
    test = dataset['test']
    idxs = np.random.choice(len(test), size=min(n_eval, len(test)), replace=False)

    # bias-only null: delta = W_U ln_f(fv)  (pure additive readout, no downstream Phi)
    ln_f, W_U = final_readout(model, model_config)
    delta = W_U(ln_f(fv.reshape(1, 1, model_config['resid_dim']).to(model.dtype)))
    delta = delta.reshape(1, -1)

    per_layer_acc, bias_acc = {}, 0.0
    for L in layers:
        correct = 0
        for j in idxs:
            sentence, target = zeroshot_prompt(test[int(j)], model_config)
            tgt_id = get_answer_id(sentence, target, tokenizer)
            tgt_id = tgt_id[0] if isinstance(tgt_id, list) else tgt_id
            clean, interv = clean_and_fv_logits(sentence, fv, L, model,
                                                model_config, tokenizer)
            if compute_individual_token_rank(interv.squeeze(), tgt_id) == 0:
                correct += 1
            if L == layers[0]:  # compute bias-null once
                if compute_individual_token_rank((clean + delta).squeeze(), tgt_id) == 0:
                    bias_acc += 1
        per_layer_acc[L] = correct / len(idxs)
    bias_acc /= len(idxs)
    return per_layer_acc, bias_acc


# ================================================================ Test 2
def make_ablation_hook(target_layer_name, mode, mean_vec=None, idx=-1):
    """edit_output hook that zeros or mean-replaces a sublayer output at token idx."""
    def fn(output, layer_name):
        if layer_name != target_layer_name:
            return output
        t = output[0] if isinstance(output, tuple) else output
        if mode == 'zero':
            t[:, idx] = 0.0
        elif mode == 'mean':
            t[:, idx] = mean_vec.to(t.dtype).to(t.device)
        return output
    return fn


@torch.no_grad()
def test2_apply_localization(dataset, fv, edit_layer, model, model_config, tokenizer,
                             n_eval=100, mode='zero'):
    """Insert FV at edit_layer; for each downstream layer l>edit_layer ablate
    attn-out vs mlp-out at the last token and measure how much FV-induced accuracy
    is destroyed. Larger drop = that component carries the applied result."""
    attn_names = model_config['attn_hook_names']
    mlp_names = get_mlp_hook_names(model_config)
    test = dataset['test']
    idxs = np.random.choice(len(test), size=min(n_eval, len(test)), replace=False)

    def fv_accuracy(extra_hook=None, extra_layer=None):
        correct = 0
        for j in idxs:
            sentence, target = zeroshot_prompt(test[int(j)], model_config)
            tgt_id = get_answer_id(sentence, target, tokenizer)
            tgt_id = tgt_id[0] if isinstance(tgt_id, list) else tgt_id
            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            fv_hook = add_function_vector(edit_layer,
                                          fv.reshape(1, model_config['resid_dim']),
                                          model.device, idx=-1)
            layers = list(model_config['layer_hook_names'])
            def combined(output, layer_name):
                output = fv_hook(output, layer_name)
                if extra_hook is not None:
                    output = extra_hook(output, layer_name)
                return output
            names = layers + ([extra_layer] if extra_layer else [])
            with TraceDict(model, layers=names, edit_output=combined):
                logits = model(**inputs).logits[:, -1, :]
            if compute_individual_token_rank(logits.squeeze(), tgt_id) == 0:
                correct += 1
        return correct / len(idxs)

    base = fv_accuracy()
    drops = {'attn': {}, 'mlp': {}}
    for l in range(edit_layer + 1, model_config['n_layers']):
        h_attn = make_ablation_hook(attn_names[l], mode)
        drops['attn'][l] = base - fv_accuracy(h_attn, attn_names[l])
        h_mlp = make_ablation_hook(mlp_names[l], mode)
        drops['mlp'][l] = base - fv_accuracy(h_mlp, mlp_names[l])
    return base, drops


# ================================================================ Test 3
@torch.no_grad()
def test3_copy_localization(dataset, model, model_config, tokenizer,
                            n_probe=32, n_icl=5, eps=0.5):
    """Cross-position mixed second difference of each sublayer's output at the
    query position, w.r.t. perturbing the input embedding at two distinct source
    token positions a != b. Attention (softmax normalization + value mixing)
    couples positions -> nonzero; a position-wise MLP generates no cross-position
    term of its own.

    CAVEAT (real model): the MLP-out captured here still INHERITS cross-position
    structure via the residual stream feeding it, so its mixed term is not zero in
    a deep net. This function therefore reports the RATIO attn/mlp and the per-layer
    trend; for the exact position-wise isolation (MLP own-term == machine zero) see
    the toy model eval_apply_probes.py, where each sublayer is evaluated on its own
    local input. Treat Test 3 as corroborating direction, Probe A (toy) as the clean
    separation.
    """
    attn_names = model_config['attn_hook_names']
    mlp_names = get_mlp_hook_names(model_config)
    d = model_config['resid_dim']
    emb = model.get_input_embeddings()
    res = {l: {'attn': 0.0, 'mlp': 0.0} for l in range(model_config['n_layers'])}

    train = dataset['train']
    for _ in range(n_probe):
        wp = train[np.random.choice(len(train), n_icl, replace=False)]
        q = train[np.random.choice(len(train), 1, replace=False)]
        q = {k: (v[0] if isinstance(v, list) else v) for k, v in q.items()}
        pd = word_pairs_to_prompt_data(wp, query_target_pair=q,
                                       prepend_bos_token=model_config['prepend_bos'])
        sentence = create_prompt(pd)
        ids = tokenizer(sentence, return_tensors='pt').to(model.device).input_ids
        T = ids.shape[1]; qpos = T - 1
        a, b = 1, 3                                  # two distinct source positions
        pa = torch.zeros(1, T, d, device=model.device); pa[0, a] = eps * torch.randn(d, device=model.device)
        pb = torch.zeros(1, T, d, device=model.device); pb[0, b] = eps * torch.randn(d, device=model.device)

        def run(pert):
            def emb_hook(output, layer_name):
                return output + pert.to(output.dtype)
            names = attn_names + mlp_names + [_emb_name(model_config)]
            with TraceDict(model, layers=names, retain_output=True,
                           edit_output=lambda o, n: emb_hook(o, n) if n == _emb_name(model_config) else o) as td:
                model(ids)
            out = {}
            for l in range(model_config['n_layers']):
                at = td[attn_names[l]].output
                mt = td[mlp_names[l]].output
                at = at[0] if isinstance(at, tuple) else at
                mt = mt[0] if isinstance(mt, tuple) else mt
                out[l] = {'attn': at[0, qpos].float().clone(),
                          'mlp': mt[0, qpos].float().clone()}
            return out

        f00 = run(torch.zeros(1, T, d, device=model.device))
        f10 = run(pa); f01 = run(pb); f11 = run(pa + pb)
        for l in range(model_config['n_layers']):
            for c in ('attn', 'mlp'):
                mixed = f11[l][c] - f10[l][c] - f01[l][c] + f00[l][c]
                res[l][c] += mixed.norm().item()
    for l in res:
        for c in res[l]:
            res[l][c] /= n_probe
    return res


def _emb_name(model_config):
    name = model_config['name_or_path'].lower()
    if 'gpt2' in name or 'gpt-j' in name:
        return 'transformer.wte'
    if 'gpt-neox' in name or 'pythia' in name:
        return 'gpt_neox.embed_in'
    return 'model.embed_tokens'


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='gpt2-xl')
    ap.add_argument('--task', default='antonym')     # any dataset_files/ task
    ap.add_argument('--edit_layer', type=int, default=9)  # ~L* for gpt2-xl antonym
    ap.add_argument('--n_top_heads', type=int, default=10)
    ap.add_argument('--n_eval', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tests', default='1,2,3')
    args = ap.parse_args()

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name,
                                                                  device=args.device)
    dataset = load_dataset(args.task, seed=0)

    print(f"[{args.model_name} | task={args.task}] extracting function vector ...")
    fv, top_heads = extract_fv(dataset, model, model_config, tokenizer,
                               n_top_heads=args.n_top_heads)
    print("top heads (L,H,AIE):", top_heads)

    todo = set(args.tests.split(','))
    if '1' in todo:
        print("\n=== Test 1: reuse + realization (edit-layer sweep) ===")
        acc, bias_acc = test1_reuse_and_realization(dataset, fv, model, model_config,
                                                     tokenizer, n_eval=args.n_eval)
        for L, a in acc.items():
            print(f"  edit_layer {L:2d}: FV-insertion zero-shot acc = {a:.3f}")
        print(f"  bias-only null (constant W_U ln_f(FV) shift) acc = {bias_acc:.3f}")
        print("  interpretation: mid-layer peak >> bias-only  =>  one fixed point,")
        print("  reused across all queries, is realized+applied downstream (Phi+eval),")
        print("  not a query-independent logit bias.")

    if '2' in todo:
        print(f"\n=== Test 2: apply localization (ablate downstream of L={args.edit_layer}) ===")
        base, drops = test2_apply_localization(dataset, fv, args.edit_layer, model,
                                               model_config, tokenizer, n_eval=args.n_eval)
        print(f"  FV-insertion base acc = {base:.3f}")
        for l in sorted(drops['attn']):
            print(f"  layer {l:2d}: attn-ablate drop = {drops['attn'][l]:+.3f}"
                  f"   mlp-ablate drop = {drops['mlp'][l]:+.3f}")
        print("  larger MLP drops => MLP realizes apply/Phi; larger attn drops => eval in attention.")

    if '3' in todo:
        print("\n=== Test 3: copy localization (cross-position interaction) ===")
        res = test3_copy_localization(dataset, model, model_config, tokenizer)
        for l in sorted(res):
            a, m = res[l]['attn'], res[l]['mlp']
            print(f"  layer {l:2d}: attn={a:.4f}  mlp={m:.4f}  ratio={a/max(1e-9,m):.2f}")
        print("  attn-dominant cross-position term localizes the (x)/copy in softmax;")
        print("  see eval_apply_probes.py (toy) for the exact MLP-own-term == 0 isolation.")


if __name__ == '__main__':
    main()
