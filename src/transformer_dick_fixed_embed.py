from __future__ import annotations
import re, ast
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import Recursive_Ttansformer as RT
from sexpdata import Symbol, dumps, loads

# =========================================================
# 1) S式 ↔ Dyck（括弧列＋ラベル列）  ※前回の往復のうち利用部
# =========================================================

def sexp_to_dyck_and_labels(sexp: Any) -> Tuple[str, List[str]]:
    dyck_chars: List[str] = []
    labels: List[str] = []
    def visit(x: Any):
        dyck_chars.append("(")
        if isinstance(x, list):
            labels.append("LIST")
            for c in x:
                visit(c)
        elif isinstance(x, Symbol):
            labels.append(f"SYM:{str(x)}")
        elif isinstance(x, bool):
            labels.append(f"BOOL:{x}")
        elif isinstance(x, (int, float)):
            labels.append(f"NUM:{repr(x)}")
        elif x is None:
            labels.append("NONE")
        else:
            raise TypeError(f"Unsupported atom: {type(x)}")
        dyck_chars.append(")")
    visit(sexp)
    return "".join(dyck_chars), labels

def sexp_str_to_dyck_and_labels(sexp_str: str) -> Tuple[str, List[str]]:
    return sexp_to_dyck_and_labels(loads(sexp_str))

# =========================================================
# 2) トークナイザ（sexp / dyck の2モード）
#    - 数値トークンは <NUM> / L:NUM に正規化し、別途 float 値を保持
# =========================================================

NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

def tokenize_sexp(s: str) -> Tuple[List[str], List[Optional[float]]]:
    s = s.replace("(", " ( ").replace(")", " ) ")
    toks_raw = s.split()
    toks: List[str] = []
    vals: List[Optional[float]] = []
    for t in toks_raw:
        if t in ("(", ")"):
            toks.append(t); vals.append(None)
        elif NUM_RE.match(t):
            toks.append("<NUM>"); vals.append(float(t))
        else:
            toks.append(t); vals.append(None)
    return toks, vals

def tokenize_dyck(sexp_str: str) -> Tuple[List[str], List[Optional[float]]]:
    dyck, labels = sexp_str_to_dyck_and_labels(sexp_str)
    dyck = re.sub(r"\s+", "", dyck)
    toks: List[str] = []
    vals: List[Optional[float]] = []
    it = iter(labels)
    for ch in dyck:
        if ch == "(":
            lab = next(it)
            toks.append("("); vals.append(None)
            if lab.startswith("NUM:"):
                toks.append("L:NUM")
                vals.append(ast.literal_eval(lab[4:]))
            else:
                toks.append(f"L:{lab}")
                vals.append(None)
        elif ch == ")":
            toks.append(")"); vals.append(None)
        else:
            raise ValueError("Dyck contains non-paren")
    return toks, vals

# =========================================================
# 3) 語彙・ID化
# =========================================================

PAD = "<PAD>"
CLS = "<CLS>"

class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = [PAD, CLS]
        seen = set(uniq)
        for t in tokens:
            if t not in seen:
                uniq.append(t); seen.add(t)
        self.stoi = {t:i for i,t in enumerate(uniq)}
        self.itos = {i:t for t,i in self.stoi.items()}
    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi.get(t, 0)) for t in toks]
    def __len__(self): return len(self.stoi)

# =========================================================
# 4) PyTorch Dataset / Collate
# =========================================================

class ExprDataset(Dataset):
    def __init__(self, samples: List[SexpSample], mode: str = "sexp"):
        self.samples = samples
        self.mode = mode
        # 事前に全トークンを集めて語彙を作る（数値は正規化）
        all_tokens: List[str] = []
        for s in samples:
            if mode == "sexp":
                toks, _ = tokenize_sexp(s.sexp)
            elif mode == "dyck":
                toks, _ = tokenize_dyck(s.sexp)
            else:
                raise ValueError("mode must be 'sexp' or 'dyck'")
            all_tokens.extend(toks)
        self.vocab = Vocab(all_tokens)

        # 数値埋め込みの正規化用（train側で再設定）
        self.num_mean = 0.0
        self.num_std  = 1.0
        self.y_mean   = 0.0
        self.y_std    = 1.0

    def set_norms(self, num_mean, num_std, y_mean, y_std):
        self.num_mean, self.num_std = num_mean, max(num_std, 1e-6)
        self.y_mean, self.y_std = y_mean, max(y_std, 1e-6)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        toks, vals = tokenize_sexp(s.sexp) if self.mode == "sexp" else tokenize_dyck(s.sexp)
        # 先頭に CLS
        toks = [CLS] + toks
        vals = [None] + vals
        ids  = self.vocab.encode(toks)

        # 数値ベクトルとマスク
        num_vals = [0.0 if v is None else ( (v - self.num_mean) / self.num_std ) for v in vals]
        num_mask = [0 if v is None else 1 for v in vals]

        # 目的変数（正規化）
        y = (self.samples[idx].value - self.y_mean) / self.y_std
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "num_vals":  torch.tensor(num_vals, dtype=torch.float),
            "num_mask":  torch.tensor(num_mask, dtype=torch.long),
            "y":         torch.tensor([y], dtype=torch.float),
            "len":       torch.tensor(len(ids), dtype=torch.long),
        }

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
# 5) Transformer 回帰モデル
#    - token embedding + pos embedding + (numeric MLP embedding × mask)
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 1024, max_len: int = 4096, dropout: float = 0.1,pad_id:int=-1):
        super().__init__()
        self.vocab_size=vocab_size
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.pad_id=pad_id
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
        num_vals: torch.Tensor | None = None,
        num_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None, ):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(input_ids) + self.pos(pos_ids)

        # numeric embedding（非数値位置は0）
        if (num_vals is not None) and (num_mask is not None):
            num_embed = self.num_proj(num_vals.unsqueeze(-1))
            num_embed = num_embed * num_mask.unsqueeze(-1).float()
            x = x + num_embed
        # --- attn_mask も省略可能にしたい場合 ---
        if attn_mask is None:
            key_padding_mask = (input_ids == self.pad_id)
        else:
            key_padding_mask = (attn_mask == 0) # True=padding

        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]  # 先頭が <CLS>
        yhat = self.head(cls)  # (B,1)
        return yhat

