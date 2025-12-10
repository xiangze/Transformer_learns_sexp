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

# =========================================================
# Transformer 回帰モデル
#    - token embedding + pos embedding + (numeric MLP embedding × mask)
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 1024, max_len: int = 4096, dropout: float = 0.1,pad_id:int=-1,debug=False):
        super().__init__()
        self.vocab_size=vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.pad_id=pad_id
        self.debug=debug
        self.num_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
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
            except:
                print("input",input_ids)
                print("max input",torch.max(input_ids))
                print("vocab_size",self.vocab_size)
                exit()
            print("shape x",x.shape)
            util.nanindex({"x":x,"input_ids":input_ids,"attn_mask":attn_mask},"x")
        else:
            x = self.tok(input_ids) + self.pos(pos_ids)
        
        # --- attn_mask も省略可能にしたい場合 ---
        if attn_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        else:
            key_padding_mask = (attn_mask == 0) # True=padding

        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        if(self.debug):
            util.nanindex({"h":h,"padding_mask":key_padding_mask},"h")
        cls = h[:, 0, :]  # 先頭が <CLS>
        yhat = self.head(cls)  # (B,1)
        return yhat

