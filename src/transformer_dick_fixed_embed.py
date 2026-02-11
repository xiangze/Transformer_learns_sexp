from __future__ import annotations
import re, ast
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import Recursive_Ttansformer as RT
from sexpdata import Symbol, dumps, loads
import util
from torch.nn.functional import _in_projection_packed

# =========================================================
# PyTorch Dataset / Collate
# =========================================================
def collate(batch):
    maxlen = max(int(b["len"]) for b in batch)
    pad_id = 0  # <PAD> は 0
    def pad1(x, pad_value):
        out = torch.full((len(batch), maxlen), pad_value, dtype=x.dtype)
        for i,b in enumerate(batch):
            L = int(b["len"])
            out[i,:L] = b[x][:L]
        return out

    input_ids = pad1("input_ids", pad_id)
    num_vals  = pad1("num_vals", 0.0)
    num_mask  = pad1("num_mask", 0)
    attn_mask = (input_ids != pad_id).long()
    y         = torch.cat([b["y"] for b in batch], dim=0)  # (B,1)
    return {"input_ids": input_ids, "num_vals": num_vals, "num_mask": num_mask,
            "attn_mask": attn_mask, "y": y}

class ForceWeightsMHA(nn.Module):
    """
    nn.MultiheadAttention をラップして、
    forward 呼び出し時に必ず need_weights=True にする。
    """
    def __init__(self, mha: nn.MultiheadAttention, average_attn_weights: bool = False):
        super().__init__()
        self.mha = mha
        self.average_attn_weights = average_attn_weights
        self.last_attn = None  # (B, H, T, S) など
        # TransformerEncoderLayer が参照することがある属性を明示的に持たせる
        self.batch_first = getattr(mha, "batch_first", False)
        
    def forward(self, query, key, value, **kwargs):
        # EncoderLayer は need_weights=False で呼んでくるので上書き
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = self.average_attn_weights
        out, attn = self.mha(query, key, value, **kwargs)
        self.last_attn = attn
        return out, attn

    def __getattr__(self, name):
        # nn.Module の属性解決を壊さないためのガード
        if name in ("mha", "average_attn_weights", "last_attn", "batch_first"):
            return super().__getattr__(name)
        # その他の属性は元の MultiheadAttention に委譲
        return getattr(self.mha, name)

def attach_encoder_attn_hooks(
    encoder: nn.TransformerEncoder,
    average_attn_weights: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """
    encoder 内の各 layer.self_attn を ForceWeightsMHA に差し替え、
    forward hook で attention weights を回収する。
    """
    attn_by_layer: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for i, layer in enumerate(encoder.layers):
        # self_attn を差し替え
        if isinstance(layer.self_attn, nn.MultiheadAttention):
            layer.self_attn = ForceWeightsMHA(layer.self_attn, average_attn_weights=average_attn_weights)

        def make_hook(layer_name: str):
            def hook(module, inputs, outputs):
                # outputs は (attn_output, attn_weights)
                _, attn = outputs
                attn_by_layer[layer_name] = attn  # 必要なら detach/cpu
            return hook
        handles.append(layer.self_attn.register_forward_hook(make_hook(f"layer{i}.self_attn")))

    return attn_by_layer, handles

def attach_all_encoder_attn_hooks(
    model: nn.Module,
    average_attn_weights: bool = False,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    """
    model 内に存在する nn.TransformerEncoder を全て見つけて
    attach_encoder_attn_hooks を適用する。

    Returns:
      attn_maps_by_encoder:
        {
          "<module_path_to_encoder>": {
             "layer0.self_attn": attn_tensor,
             ...
          },
          ...
        }
      handles: hook の handle 一式（後で remove する用）
    """
    attn_maps_by_encoder: Dict[str, Dict[str, torch.Tensor]] = {}
    handles_all: List[torch.utils.hooks.RemovableHandle] = []

    for module_path, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoder):
            attn_by_layer, handles = attach_encoder_attn_hooks(
                module, average_attn_weights=average_attn_weights, )
            attn_maps_by_encoder[module_path] = attn_by_layer
            handles_all.extend(handles)

    return attn_maps_by_encoder, handles_all

@torch.no_grad()
def compute_qk(mha: torch.nn.MultiheadAttention, x: torch.Tensor):
    """
    mha: nn.MultiheadAttention
    x : (B,T,E) if batch_first=True else (T,B,E)
    returns:
      q, k: (B,H,T,Hd)
      logits: (B,H,T,T)  where logits = (q @ k^T) / sqrt(Hd)
    """
    # ---- shape unify to (B,T,E)
    if mha.batch_first:
        x_btE = x
    else:
        x_btE = x.transpose(0, 1)  # (B,T,E)

    B, T, E = x_btE.shape
    H = mha.num_heads
    assert E % H == 0, f"d_model={E} must be divisible by num_heads={H}"
    Hd = E // H

    # ---- project to q,k (and v not needed)
    # Standard MHA uses a packed projection: [Wq; Wk; Wv] in in_proj_weight
    W = mha.in_proj_weight          # (3E, E)
    b = mha.in_proj_bias            # (3E) or None
    qkv = F.linear(x_btE, W, b)     # (B,T,3E)
    q, k, _v = qkv.split(E, dim=-1) # each (B,T,E)

    # ---- reshape to (B,H,T,Hd)
    q = q.view(B, T, H, Hd).transpose(1, 2).contiguous()
    k = k.view(B, T, H, Hd).transpose(1, 2).contiguous()

    # ---- logits: (B,H,T,T)
    qk=q @ k.transpose(-2, -1)
    #logits = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
    return q, k, qk

def attach_qk_hooks_to_transformer_encoder(model: torch.nn.Module):
    """
    model: nn.TransformerEncoder でも encoder を含む任意のモデルでもOK
    戻り値:
      cache: 取り出した logits などを溜める辞書
      handles: remove() 用の handle 一覧
    """
    cache = {}
    handles = []

    # TransformerEncoder なら通常 model.layers に EncoderLayer が入っている
    # それ以外でも "self_attn" を持つモジュールを探索して貼るのが堅い
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            cache[name] = {"q": [], "k": [], "logits": []}

            def make_hook(mha_name):
                def pre_hook(mha_module, args):
                    # MultiheadAttention.forward(self, query, key, value, ...)
                    query = args[0]
                    # self-attention前提で query==key==value のはず
                    q, k, logits = compute_qk(mha_module, query)
                    cache[mha_name]["q"].append(q.cpu())
                    cache[mha_name]["k"].append(k.cpu())
                    cache[mha_name]["logits"].append(logits.cpu())
                return pre_hook

            h = mod.register_forward_pre_hook(make_hook(name))
            handles.append(h)

    return cache, handles


class EncoderLayerWithQK(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # src: (B, T, E) if batch_first=True
        x = src
        attn = self.self_attn
        assert attn.batch_first, "この例は batch_first=True 前提です"

        # --- Q,K を取り出す（MultiheadAttention の投影と同じ重みを使う） ---
        # PyTorch実装に合わせて in_proj_weight/bias を使う
        q, k, v = _in_projection_packed(x, x, x, attn.in_proj_weight, attn.in_proj_bias)

        B, T, E = q.shape
        H = attn.num_heads
        Hd = E // H
        # (B, H, T, Hd)
        q = q.view(B, T, H, Hd).transpose(1, 2)
        k = k.view(B, T, H, Hd).transpose(1, 2)

        # --- softmax前のスコア（logits）: (B, H, T, T) ---
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)

        # 参考：mask を適用したいならここで（可視化目的なら未適用の方が見やすいことも多い）
        if src_mask is not None:
            # src_mask は (T,T) or (B*H,T,T) などの場合があるので用途に合わせて調整
            pass
        if src_key_padding_mask is not None:
            # (B,T) Trueがpad なら logits[..., pad_pos] を -inf にする等
            pass

        # --- 通常の TransformerEncoderLayer の処理（= 出力も返す） ---
        # 元実装に近い形で self_attn を呼んで出力を得る
        attn_out = attn(x, x, x,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        need_weights=False,
                        is_causal=is_causal)[0]
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x, {"q": q, "k": k, "logits": logits}

# =========================================================
# Transformer 回帰モデル
#    - token embedding + pos embedding + (numeric MLP embedding × mask)
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, params:dict,debug=False,outQK=False):
        super().__init__()
        vocab_size=params["vocab_size"]
        d_model=params["d_model"]
        nhead=params["nhead"]
        num_layers = params["num_layer"]
        dim_ff= params["dim_ff"]
        max_len= params["max_len"]
        self.pad_id=params["pad_id"]
        dropout=params["dropout"]
        self.vocab_size=vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.debug=debug
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        if(outQK):
            self.enc = EncoderLayerWithQK(nn.TransformerEncoderLayer)
        else:
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max_len)
        )


    def forward(self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None, ):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        if(self.debug):
            try:
                x = self.tok(input_ids) + self.pos(pos_ids)
                print("tok",self.tok.weight.detach().cpu())
                print("pos",self.pos.weight.detach().cpu())
            except:
                print("shape",self.tok(input_ids).shape)
                print("pos",pos_ids.shape)
                print("pos",pos_ids)
                print("max input",torch.max(input_ids))
                print("vocab_size",self.vocab_size)
                exit()
        else:
            x = self.tok(input_ids) + self.pos(pos_ids)  #a

        if attn_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        else:
            key_padding_mask = (attn_mask == 0) # True=padding
        key_padding_mask[:, 0] = False

        valid = (~key_padding_mask).sum(dim=1)
        assert torch.all(valid > 0), "Some sequences fully masked!"

        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]  # 先頭が <CLS>
        yhat = self.head(cls)  # (B,1)
        if(self.debug):
            util.nanindex({"h":h,"padding_mask":key_padding_mask,"x":x,"cls":cls,"yhat":yhat},"h")
        return yhat
    
    def add_hook(self, average_attn_weights=False):
        self.attn_dict, self.hooks = attach_encoder_attn_hooks(self.enc, average_attn_weights)        
        return self.attn_dict, self.hooks
if __name__ == "__main__":
    import torch.optim as optim
    params = {
        "vocab_size": 4,
        "d_model": 4,
        "nhead": 1,
        "num_layer": 1,
        "dim_ff": 2,
        "max_len": 50,
        "pad_id": 0,
        "dropout": 0., }
    B = 2
    L = 10
    N = 20
    for mask in [False,True]:
        model = TransformerRegressor(params,debug=True)
        optimizer=optim.Adam(model.parameters(), lr=0.05)
        optimizer.zero_grad(set_to_none=True)
        #1 epoch
        model.train()
        if(mask):
            print("with mask")
            attn_mask = torch.ones((B, L), dtype=torch.long)
            attn_mask[0, 5:] = 0  # padding
        else:
            print("without mask")
        for i in range(N):
            input_ids = torch.randint(0, params["vocab_size"], (B, L))
            label=input_ids.sum(dim=1,keepdim=True).float()
            if(mask):
                yhat=model(input_ids,attn_mask)
                valid_mask = (1-attn_mask).unsqueeze(-1).float()  # (B,L,1)
                loss_raw = (yhat - label)**2
                print("loss",loss_raw.shape,"mask",valid_mask.shape)
                loss = (loss_raw * valid_mask.squeeze(-1)).sum()/valid_mask.sum() ##
            else:
                yhat=model(input_ids,None)
                loss=((yhat - label)**2).sum()
            loss.backward()
            optimizer.step()
            print("yhat:",yhat)
            print("loss:", loss)

