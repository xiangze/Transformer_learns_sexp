from __future__ import annotations
import re, ast
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import Recursive_Ttansformer as RT
from sexpdata import Symbol, dumps, loads
import util
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

    def forward(self, query, key, value, **kwargs):
        # EncoderLayer は need_weights=False で呼んでくるので上書き
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = self.average_attn_weights

        out, attn = self.mha(query, key, value, **kwargs)
        self.last_attn = attn
        return out, attn


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

        name = f"layer{i}.self_attn"
        def make_hook(layer_name: str):
            def hook(module, inputs, outputs):
                # outputs は (attn_output, attn_weights)
                _, attn = outputs
                attn_by_layer[layer_name] = attn  # 必要なら detach/cpu
            return hook
        handles.append(layer.self_attn.register_forward_hook(make_hook(name)))

    return attn_by_layer, handles

# =========================================================
# Transformer 回帰モデル
#    - token embedding + pos embedding + (numeric MLP embedding × mask)
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, params:dict,debug=False):
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
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max_len)
        )
        attn_dict, hooks = attach_encoder_attn_hooks(self.encoder, average_attn_weights=False)

    def forward(self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None, ):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        if(self.debug):
            try:
                x = self.tok(input_ids) + self.pos(pos_ids)
            except:
                print("input",input_ids)
                print("max input",torch.max(input_ids))
                print("vocab_size",self.vocab_size)
                exit()
        else:
            x = self.tok(input_ids) + self.pos(pos_ids)

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
                loss = (loss_raw * valid_mask.squeeze(-1)).sum()/valid_mask.sum()
            else:
                yhat=model(input_ids,None)
                loss=((yhat - label)**2).sum()
            loss.backward()
            optimizer.step()
            print("yhat:",yhat)
            print("loss:", loss)


