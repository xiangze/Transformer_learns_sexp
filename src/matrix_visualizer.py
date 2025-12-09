import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer, AutoConfig

def plot_heatmap(matrix, xlabels, ylabels, title, out_path):
    plt.figure(figsize=(max(6, len(xlabels)*0.5), max(5, len(ylabels)*0.5)))
    plt.imshow(matrix, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xticks(range(len(xlabels)), xlabels, rotation=90)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_param_matrix(param_tensor, title, out_path, max_dim=1024):
    # 2次元の重み行列だけ可視化（巨大行列は切り出し）
    if param_tensor.ndim != 2:
        return False
    W = param_tensor.detach().cpu()
    if W.shape[0] > max_dim or W.shape[1] > max_dim:
        W = W[:max_dim, :max_dim]
        title = f"{title} (clipped to {max_dim}x{max_dim})"
    plt.figure(figsize=(6,5))
    plt.imshow(W, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True

def print_and_dump_attention_params(model, out_dir, layer_hint=None):
    """
    モデル内の注意関連パラメータ名を列挙し、Q/K/V らしきものを画像保存（行列のみ）。
    layer_hint が与えられたら、その層番号に関係しそうな名前を優先表示。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== [Parameter listing: attention related] ===")
    candidates = ("attn", "attention", "q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")
    plotted = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if any(tok in low for tok in candidates):
            if (layer_hint is not None) and (f".{layer_hint}." not in low and f"layer.{layer_hint}." not in low):
                # 可能なら該当層を優先。全出力したい場合はこの if を外す
                pass
            print(f"{name:60s} {tuple(p.shape)}")
            # 可視化（2Dのみ）
            safe_name = name.replace(".", "_").replace("/", "_")
            png = out_dir / f"param_{safe_name}.png"
            if save_param_matrix(p, name, png):
                plotted += 1
    print(f"Saved {plotted} attention-related weight heatmaps to: {out_dir}")

def split_gpt2_c_attn_weight(weight, hidden_size):
    """
    GPT-2 の c_attn.weight は [hidden, 3*hidden] で QKV が結合されている。
    これを (Wq, Wk, Wv) に分割する。
    """
    W = weight.detach().cpu()
    assert W.shape[1] == 3 * hidden_size
    Wq = W[:, :hidden_size]
    Wk = W[:, hidden_size:2*hidden_size]
    Wv = W[:, 2*hidden_size:]
    return Wq, Wk, Wv

def get_token_strings(tokenizer, input_ids):
    # special tokens をそのまま表示（必要ならフィルタリング可）
    return tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

def run_hf_attention_vis(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model, config=config)
    model.eval().to(device)

    # 推論時に attention を出してもらう
    # encoder-decoder では encoder_attentions / decoder_attentions / cross_attentions が得られる
    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    # attentions の在処を統一的に取得
    attn_packs = {}
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        attn_packs["self/encoder_or_causal"] = outputs.attentions
    if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
        attn_packs["encoder_self"] = outputs.encoder_attentions
    if hasattr(outputs, "decoder_attentions") and outputs.decoder_attentions is not None:
        attn_packs["decoder_self"] = outputs.decoder_attentions
    if hasattr(outputs, "cross_attentions") and outputs.cross_attentions is not None:
        attn_packs["cross"] = outputs.cross_attentions

    if not attn_packs:
        raise RuntimeError("This model didn't return attentions. Try a different model or architecture.")

    tokens = get_token_strings(tokenizer, inputs["input_ids"])

    for pack_name, attns in attn_packs.items():
        # attns: list over layers, each [batch, heads, tgt_len, src_len]
        num_layers = len(attns)
        print(f"\n[{pack_name}] layers={num_layers}, shape[0]={tuple(attns[0].shape)}")
        target_layer = args.layer if args.layer is not None else 0
        if not (0 <= target_layer < num_layers):
            raise ValueError(f"--layer must be in [0, {num_layers-1}] for pack '{pack_name}'")

        layer_attn = attns[target_layer][0]  # [heads, tgt_len, src_len]
        num_heads = layer_attn.shape[0]
        print(f"Layer {target_layer}: heads={num_heads}, tgt_len={layer_attn.shape[1]}, src_len={layer_attn.shape[2]}")

        if args.avg_heads:
            A = layer_attn.mean(dim=0).cpu().numpy()  # [tgt, src]
            title = f"{pack_name} | layer {target_layer} | head=AVG"
            out_path = out_dir / f"attn_{pack_name}_layer{target_layer}_avg.png"
            plot_heatmap(A, tokens, tokens, title, out_path)
            print(f"Saved {out_path}")
        else:
            head = args.head if args.head is not None else 0
            if not (0 <= head < num_heads):
                raise ValueError(f"--head must be in [0, {num_heads-1}] for pack '{pack_name}'")
            A = layer_attn[head].cpu().numpy()  # [tgt, src]
            title = f"{pack_name} | layer {target_layer} | head {head}"
            out_path = out_dir / f"attn_{pack_name}_layer{target_layer}_head{head}.png"
            plot_heatmap(A, tokens, tokens, title, out_path)
            print(f"Saved {out_path}")

    # パラメーター列挙＋可視化（Q/K/V らしきもの）
    print_and_dump_attention_params(model, out_dir / "params", layer_hint=args.layer)

    # GPT-2 専用: c_attn を分割して保存（もし該当すれば）
    # 多くの GPT 系で h.<i>.attn.c_attn.weight が存在
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if hidden_size is not None:
        for name, p in model.named_parameters():
            if name.endswith(".attn.c_attn.weight") and p.ndim == 2 and p.shape[1] == 3*hidden_size:
                layer_idx = "".join(ch for ch in name if ch.isdigit()) or "X"
                Wq, Wk, Wv = split_gpt2_c_attn_weight(p, hidden_size)
                for tag, W in zip(("Q","K","V"), (Wq, Wk, Wv)):
                    path = out_dir / f"gpt2_c_attn_layer{layer_idx}_{tag}.png"
                    save_param_matrix(W, f"{name} -> {tag}", path)
                print(f"Saved split Q/K/V for {name} to {out_dir}")

def attach_hooks_for_vanilla_transformer(model):
    """
    PyTorch nn.MultiheadAttention に forward hook を付けて attention weights を捕捉。
    返り値: captured dict {module_name: last_weights_tensor}
    """
    captured = {}

    def make_hook(name):
        def hook(mod, inp, out):
            # out: (attn_output, attn_output_weights) の可能性が高い
            if isinstance(out, tuple) and len(out) >= 2:
                captured[name] = out[1].detach().cpu()  # [tgt_len, src_len] or with batch dims
            else:
                # PyTorch version により異なり得る
                captured[name] = out
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.register_forward_hook(make_hook(name))
    return captured

def save_vanilla_attention_heatmap(model,tokenlength,out_dir,pname):
    captured = attach_hooks_for_vanilla_transformer(model)
    tokens = [f"t{i}" for i in range(tokenlength)]
    for name, W in captured.items():
        # 形状整形：MultiheadAttention の重みは [batch*nheads, tgt, src] など版差ありうる
        A = W
        while A.ndim < 2:  # 念のため
            A = A.unsqueeze(0)
        if A.ndim == 2:
            plot_heatmap(A.numpy(), tokens, tokens, f"[vanilla] {name}",f"{out_dir}/vanilla_{pname}_{name}.png")
            print(f"Saved vanilla attention heatmap(s) for {name}")
        elif A.ndim == 3:
            # [heads?, tgt, src] と仮定してヘッド平均
            Aavg = A.mean(dim=0).numpy()
            plot_heatmap(Aavg, tokens, tokens, f"[vanilla] {name} (avg heads)", f"{out_dir}/vanilla_{pname}_{name}_avg.png")
            print(f"Saved vanilla attention heatmap(s) for {name}")
        else:
            print("cannot save heatmap")

def main():
    parser = argparse.ArgumentParser(description="Read & visualize parameters and attention matrices from pretrained Transformers (Hugging Face).")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or path (e.g., bert-base-uncased, gpt2, cl-tohoku/bert-base-japanese-v3)")
    parser.add_argument("--text", type=str, required=True, help="Input text to compute attentions for")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to visualize (default: 0)")
    parser.add_argument("--head", type=int, default=None, help="Head index to visualize (default: 0)")
    parser.add_argument("--avg_heads", action="store_true", help="Average over heads instead of selecting a head")
    parser.add_argument("--out", type=str, default="attn_out", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--vanilla_demo", action="store_true",help="(Optional) Run a tiny demo with torch.nn.TransformerEncoder and hooks instead of HF model")
    args = parser.parse_args()

    if args.vanilla_demo:
        # 小さなデモ：nn.TransformerEncoder で hooks により注意重みを取得
        d_model, nhead, dim_ff, nlayers = 64, 4, 128, 2
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        captured = attach_hooks_for_vanilla_transformer(model)

        # ダミー入力（トークン列長さ=10）
        x = torch.randn(1, 10, d_model)
        src_key_padding_mask = None
        # 実行（need_weights=True は内部で指定済みの実装差があるので hook ベース）
        with torch.no_grad():
            _ = model(x, mask=None, src_key_padding_mask=src_key_padding_mask)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_vanilla_attention_heatmap(model,10,out_dir)
        return

    # HuggingFace モデルでの本処理
    run_hf_attention_vis(args)

if __name__ == "__main__":
    main()
