import argparse
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple
import transformer_dick_fixed_embed as tr
import seaborn as sns
import numpy as np

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

def plot_multi_attention_heatmaps(attn_maps_by_encoder:dict,params,out_dir="./",pname="",show=False):
    assert(len(attn_maps_by_encoder)>0)
    xs=[]
    for e,(enc_path, attn_by_layer) in enumerate(attn_maps_by_encoder.items()):
        print("Encoder:", enc_path)
        for i,(layer_name, attn) in enumerate(attn_by_layer.items()):
            assert(attn.dim() == 4)
            B, H, T, S = attn.shape
            if(show):
                print(layer_name, attn.shape)
            xs.append([])
            for h in range(H):
                x=attn[0, h].detach().cpu()
                xs[i].append(x)
                if(show):
                    print(f"head:{h},min value:{float(x.min())},max value:{float(x.max())},NaN {torch.isnan(x).any().item()},Inf {torch.isinf(x).any().item()}")

    fig,axes=plt.subplots(params["num_layer"],params["nhead"], figsize=(4*params["nhead"],4*params["num_layer"]), sharex=True)
    axes = np.atleast_2d(axes)

    global_min=1e9
    global_max=0
    xs=[[xi.numpy() for xi in x ]for x in xs]
    for i,x in enumerate(xs):
        global_max=max(global_max,np.max(x))
        global_min=min(global_min,np.min(x))
        for h,xi in enumerate(x):
            ax = axes[i, h]
            im=ax.imshow(xi,cmap="viridis",aspect="auto", vmin=global_min,vmax=global_max)        
            ax.set_title(f"Head {h}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            ax.grid(True)
    assert global_min<=global_max, f"xs {xs}"
    fig.suptitle(pname)
    cbar = ax.figure.colorbar(im, ax=ax, **{})
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{pname}.png")
    plt.close(fig)
    xs=np.array(xs)
    np.save(f"data/{pname}.npy",xs)
    return xs

def gen_TransformerEncoder1(params):
    return torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=params["d_model"],
                                                                        nhead=params["nhead"], 
                                                                        dim_feedforward=params["dim_ff"], batch_first=True),
                                        num_layers=params["num_layer"])

def vanilla_demo(params={"d_model":64,"nhead":4,"dim_ff":128,"num_layer":2,"max_len":1024,"vocab_size":10},
                 device="cuda",x=None,mask=None,out_dir="./",pname=""):    
        # 小さなデモ：nn.TransformerEncoder で hooks により注意重みを取得
        save_attention_heatmap(gen_TransformerEncoder1(params),params,params["vocab_size"],device,pname,x,mask,out_dir,show=True)

def save_attention_heatmap(model,params:dict,vocab_size,device,pname,x=None,mask=None,out_dir="./",show=False,getAttention=True):
        model.to(device)
        model.eval()
        if(getAttention):
            attn_maps_by_encoder, handles = tr.attach_all_encoder_attn_hooks(model,average_attn_weights=False)# head ごとに取りたいなら False
            pname="Amat_"+pname
            # forward を 1 回回す（ここで辞書が埋まる）
            x=model(x,mask)
            assert(not torch.isnan(x).any())
            # torch.backends.cuda.enable_flash_sdp(False)
            # torch.backends.cuda.enable_mem_efficient_sdp(False)
            # torch.backends.cuda.enable_math_sdp(True)
            plot_multi_attention_heatmaps(attn_maps_by_encoder,params,out_dir,pname,show)
            # 終わったら hook を外す（重要）
            for h in handles:
                h.remove()
            #print(attn_maps_by_encoder)
        else:
            pname="QK"+pname
            print(pname)
            #attn_maps_by_encoder, handles = tr.attach_qk_hooks_to_transformer_encoder(model)
            x = torch.randn(1, 1, params["d_model"]).to(device)
            qks=[]
            for layer in model.enc.layers:
                x, out = layer(x)       # 単層で確認するのが楽
                qks.append(out["qk"])
                #print(layer,qks[-1])

            fig,axes=plt.subplots(params["num_layer"],params["nhead"], figsize=(4*params["nhead"],4*params["num_layer"]), sharex=True)
            axes = np.atleast_2d(axes)
            for i,x in enumerate(qks):
                ax = axes[i, h]
                im=ax.imshow(x,cmap="viridis",aspect="auto")#, vmin=global_min,vmax=global_max)        
                ax.set_title(f"layer {i}")
                ax.set_xlabel("Key")
                ax.set_ylabel("Query")
                ax.grid(True)
            fig.suptitle(pname)
            cbar = ax.figure.colorbar(im, ax=ax, **{})
            cbar.ax.set_ylabel("", rotation=-90, va="bottom")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{pname}.png")
            plt.close(fig)
            np.save(f"data/{pname}.npy",np.arrays(qks))

def extract_qkv_weights(mha: torch.nn.MultiheadAttention):
    E = mha.embed_dim
    if mha.in_proj_weight is not None:
        W = mha.in_proj_weight.detach().cpu()            # (3E, E)
        Wq, Wk, Wv = W[:E], W[E:2*E], W[2*E:3*E]
    else:
        # in_proj_weight を使わない設定（下のBケース）
        Wq = mha.q_proj_weight.detach().cpu()
        Wk = mha.k_proj_weight.detach().cpu()
        Wv = mha.v_proj_weight.detach().cpu()
    return Wq, Wk, Wv

def get_qkv_weights(model,num_heads,x=None):
    Ws=[]
    nh=num_heads
    for layer in model.layers:
        for W in extract_qkv_weights(layer.self_attn):
            if(x!=None):
                W=W@x
                WW=[W[h*(W.shape[0]//nh):(h+1)*(W.shape[0]//nh)] for h in range(nh)] 
            else:
                WW=[W[h*(W.shape[0]//nh):(h+1)*(W.shape[0]//nh), :] for h in range(nh)] 
            Ws.append(WW)
    return Ws #[ Wq, Wk, Wv] 

def show_QKV(model, pname,num_heads,out_dir,x=None,device="cuda"):
    model.to(device)
    model.eval()
    WS={"Q":[],"K":[],"V":[]}
    global_min=1e9
    global_max=0
    for l,alayer in enumerate(get_qkv_weights(model,num_heads,x)):
        for title, W in zip(["Q","K","V"],alayer):
            for Whead in W:
                Whead=Whead.numpy()
                global_max=max(global_max,np.max(Whead))
                global_min=min(global_min,np.min(Whead))
                WS[title].append(Whead)
    ln=len(WS["Q"])
    fig,axes=plt.subplots(3,ln, figsize=(4*ln,4*3), sharex=True)
    fig.suptitle(pname)
    axes = np.atleast_2d(axes)
    for i,(title,w) in enumerate(WS.items()):
        for l,wl in enumerate(w):
            ax = axes[i,l]
            im=ax.imshow(wl,cmap="viridis",aspect="auto", vmin=global_min,vmax=global_max)        
            ax.set_title(f"Layer {l}, {title}")
            ax.grid(True)
    cbar = ax.figure.colorbar(im, ax=ax, **{})
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{pname}.png")
    plt.close(fig)
        
def show_QKV_demo(params={"d_model":64,"nhead":4,"dim_ff":128,"num_layer":2,"max_len":1024,"vocab_size":10},
                 device="cuda",out_dir="./",pname="QKV",x=None):    
        show_QKV(gen_TransformerEncoder1(params),pname,params["nhead"],out_dir=out_dir,device=device,x=x)


def main():
    parser = argparse.ArgumentParser(description="Read & visualize parameters and attention matrices from pretrained Transformers (Hugging Face).")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model name or path (e.g., bert-base-uncased, gpt2, cl-tohoku/bert-base-japanese-v3)")
    parser.add_argument("--text", type=str, default=None, help="Input text to compute attentions for")
    parser.add_argument("--layer", type=int, default=2, help="Layer index to visualize (default: 0)")
    parser.add_argument("--head", type=int, default=4, help="Head index to visualize (default: 0)")
    parser.add_argument("--avg_heads", action="store_true", help="Average over heads instead of selecting a head")
    parser.add_argument("--out", type=str, default="attn_out", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--demo", action="store_true",help="(Optional) Run a tiny demo with torch.nn.TransformerEncoder and hooks instead of HF model")
    parser.add_argument("--qkv", action="store_true", help="Show QKV matrices")
    args = parser.parse_args()

    if args.demo:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        params={"d_model":64,"nhead":4,"dim_ff":128,"num_layer":2,"max_len":1024,"vocab_size":10}
        vanilla_demo(params,device,x=None,mask=None,out_dir=args.out,pname="vanilla_demo_matrix")
    elif args.qkv:
        show_QKV_demo(out_dir=args.out)
    else:
        # HuggingFace モデルでの本処理
        assert args.model!=None ,"HuggingFace Model should by set"
        assert args.text!=None , "input text should by set"
        run_hf_attention_vis(args)

if __name__ == "__main__":
    main()
