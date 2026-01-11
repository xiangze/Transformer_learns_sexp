import argparse
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import transformer_dick_fixed_embed as tr
import seaborn as sns

def plot_heatmap(matrix, xlabels, ylabels, title, out_path):
    print(f"plot heatmap {out_path}")
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


def plot_vanilla_attention_heatmap(tokenlength,attn_dict,params,out_dir="./",pname=""):
    tokens = [f"t{i}" for i in range(tokenlength)]
    for name,A in attn_dict.items():
        # 形状整形：MultiheadAttention の重みは [batch*nheads, tgt, src] など版差ありうる
        assert A.ndim<5 ,f"dimention is too large {A.shape}"
        assert A.ndim>2 ,f"dimention is too small {A.shape}"
        fname=f"{out_dir}/Amat_{pname}_{A.ndim}.png"
        if A.ndim == 2:
            plot_heatmap(A, tokens, tokens, f"{name}", fname)
        elif A.ndim == 3:   # [heads?, tgt, src] と仮定してヘッド平均
            plot_heatmap(A.mean(dim=0).numpy(), tokens, tokens, f"{name}(avg heads with averaged )",fname)
        elif A.ndim == 4:
            plot_heatmap(A[0].mean(dim=0).numpy(), tokens, tokens, f"{name}(avg heads,bacth=0)",fname)

def plot_multi_attention_heatmaps(attn_dict,params,out_dir="./",pname=""):
    fig,axes=plt.subplots(params["num_layer"],params["nhead"], figsize=(10,10), sharex=True)
    for i,(layer_name, attn) in enumerate(attn_dict.items()):
        assert(attn.dim() == 4),f"{layer_name}: unexpected shape {attn.shape}"
        B, H, T, S = attn.shape
        fig.suptitle(layer_name)
        for h in range(H):
            ax = axes[i, h]
            ax.imshow( attn[0, h].detach().cpu(), cmap="viridis",aspect="auto", vmin=0.0)#vmax
            ax.set_title(f"Head {h}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
    plt.title("Attention matrix"+pname)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Amat_{pname}.png")

def vanilla_demo(tokenlength,out_dir="./",pname=""):    
        # 小さなデモ：nn.TransformerEncoder で hooks により注意重みを取得
        d_model, nhead, dim_ff, nlayers = 64, 4, 128, 2
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        attn_dict,hooks = tr.attach_encoder_attn_hooks(model)
        # ダミー入力（トークン列長さ=10）
        x = torch.randn(1, 10, d_model)
        # 実行（need_weights=True は内部で指定済みの実装差があるので hook ベース）
        with torch.no_grad():
            _ = model(x, mask=None, src_key_padding_mask=None)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        plot_vanilla_attention_heatmap(tokenlength,attn_dict,out_dir,pname="sexp_")

def save_attention_heatmap(model,params:dict,vocab_size,device,pname,x=None,mask=None,out_dir="./"):
        tokenlength=params["max_len"] 
        attn_dict,_ = model.add_hook()
        if(x is None):
            x = torch.randint(vocab_size,(tokenlength,params["d_model"])).to(device) # ダミー入力(int)
            mask=None
        #print("x",x.dim(),x.shape)
        # 実行（need_weights=True は内部で指定済みの実装差があるので hook ベース）
        #model.eval()
        with torch.no_grad():
            model(x,mask)#, src_key_padding_mask=None)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        plot_multi_attention_heatmaps(attn_dict,params,out_dir,pname)

def main():
    parser = argparse.ArgumentParser(description="Read & visualize parameters and attention matrices from pretrained Transformers (Hugging Face).")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model name or path (e.g., bert-base-uncased, gpt2, cl-tohoku/bert-base-japanese-v3)")
    parser.add_argument("--text", type=str, default=None, help="Input text to compute attentions for")
    parser.add_argument("--layer", type=int, default=None, help="Layer index to visualize (default: 0)")
    parser.add_argument("--head", type=int, default=None, help="Head index to visualize (default: 0)")
    parser.add_argument("--avg_heads", action="store_true", help="Average over heads instead of selecting a head")
    parser.add_argument("--out", type=str, default="attn_out", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--demo", action="store_true",help="(Optional) Run a tiny demo with torch.nn.TransformerEncoder and hooks instead of HF model")
    args = parser.parse_args()

    if args.demo:
        vanilla_demo(20,out_dir="./",pname="")
    else:
        # HuggingFace モデルでの本処理
        assert args.model!=None ,"HuggingFace Model should by set"
        assert args.text!=None , "input text should by set"
        run_hf_attention_vis(args)

if __name__ == "__main__":
    main()
