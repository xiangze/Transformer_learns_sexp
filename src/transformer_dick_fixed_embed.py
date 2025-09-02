# -*- coding: utf-8 -*-
# transformer_dyck_fixed_embed.py
from __future__ import annotations
import re, ast, math, random, argparse, hashlib
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import hy
from sexpdata import Symbol, dumps, loads

# =========================================================
# 0) S式データ生成（数値式）＆ Hy評価（簡約版）
# =========================================================
_NUM_UNARY_FUNCS = ["abs", "round"]
_NUM_BINARY_OPS  = ["+", "-", "*", "/"]
_NUM_NARY_FUNCS  = ["max", "min", "sum"]
_POW_FUNC        = "pow"
_CMP_OPS         = ["<", "<=", ">", ">=", "=", "!="]

def _rand_number(int_ratio: float = 0.6) -> float | int:
    if random.random() < int_ratio:
        return random.randint(-9, 9)
    return round(random.uniform(-9.0, 9.0), 2)

def _rand_small_int(low: int = -4, high: int = 4) -> int:
    x = 0
    while x == 0:
        x = random.randint(low, high)
    return x

def _gen_numeric_expr(depth: int) -> Any:
    if depth <= 0 or random.random() < 0.35:
        return _rand_number()
    choice = random.random()
    if choice < 0.35:
        op = random.choice(_NUM_BINARY_OPS)
        a = _gen_numeric_expr(depth - 1)
        b = _gen_numeric_expr(depth - 1)
        return [Symbol(op), a, b]
    elif choice < 0.55:
        f = random.choice(_NUM_UNARY_FUNCS)
        x = _gen_numeric_expr(depth - 1)
        return [Symbol(f), x]
    elif choice < 0.7:
        base = _gen_numeric_expr(depth - 1)
        k = _rand_small_int()
        return [Symbol(_POW_FUNC), base, k]
    elif choice < 0.85:
        f = random.choice(_NUM_NARY_FUNCS)
        n_items = random.randint(2, 5)
        items = [_gen_numeric_expr(depth - 1) for _ in range(n_items)]
        return [Symbol(f), [Symbol("list"), *items]]
    else:
        cond = _gen_comparison(depth - 1)
        then_e = _gen_numeric_expr(depth - 1)
        else_e = _gen_numeric_expr(depth - 1)
        return [Symbol("if"), cond, then_e, else_e]

def _gen_comparison(depth: int) -> Any:
    op = random.choice(_CMP_OPS)
    a = _gen_numeric_expr(depth - 1)
    b = _gen_numeric_expr(depth - 1)
    return [Symbol(op), a, b]

def _eval_hy_expr_str(expr_str: str) -> Any:
    model = hy.read_str(expr_str)
    return hy.eval(model)

@dataclass
class SexpSample:
    sexp: str
    value: float

def make_sexp_eval_dataset(n_samples: int = 5000, *, max_depth: int = 4, seed: int = 0) -> List[SexpSample]:
    random.seed(seed)
    out: List[SexpSample] = []
    while len(out) < n_samples:
        sexp_ast = _gen_numeric_expr(max_depth)
        sexp_str = dumps(sexp_ast)
        try:
            val = _eval_hy_expr_str(sexp_str)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out.append(SexpSample(sexp=sexp_str, value=float(val)))
        except Exception:
            pass
    return out

# =========================================================
# 1) S式 → Dyck（括弧列＋ラベル列）
# =========================================================
@dataclass
class _Node:
    children: List["_Node"]

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
# 2) トークナイズ（Dyck + ラベル）
#    例: "(" , "L:LIST" , ")" , "(" , "L:SYM:*" , "L:NUM" , ")"
# =========================================================
NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

CLS = "<CLS>"
PAD = "<PAD>"

def tokenize_dyck(sexp_str: str) -> Tuple[List[str], List[Optional[float]]]:
    dyck, labels = sexp_str_to_dyck_and_labels(sexp_str)
    dyck = re.sub(r"\s+", "", dyck)
    toks: List[str] = []
    vals: List[Optional[float]] = []
    it = iter(labels)
    for ch in dyck:
        if ch == "(":
            lab = next(it)
            toks.append("(");        vals.append(None)
            if lab.startswith("NUM:"):
                toks.append("L:NUM"); vals.append(ast.literal_eval(lab[4:]))
            else:
                toks.append(f"L:{lab}"); vals.append(None)
        elif ch == ")":
            toks.append(")");         vals.append(None)
    return toks, vals

# =========================================================
# 3) 固定写像（関数的エンコーディング） or 学習埋め込み
# =========================================================
def stable_hash_u32(s: str) -> int:
    # 実行毎に変わらない安定ハッシュ
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

def token_to_fixed_vec(tok: str, d_model: int) -> torch.Tensor:
    """
    学習なしの固定特徴ベクトルに写像。
    先頭次元に括弧の別、次に代表カテゴリ（NUM/LIST/SYM/BOOL/NONE/CLS）、
    余った次元で安定ハッシュによるワンホット（衝突回避のため少数次元）。
    """
    v = torch.zeros(d_model)
    # 0/1: 括弧
    if tok == "(":
        v[0] = 1.0
    elif tok == ")":
        v[1] = 1.0

    # 2..: カテゴリフラグ
    base = 2
    def set_bit(i): 
        if base+i < d_model: v[base+i] = 1.0

    if tok == CLS:
        set_bit(0)  # CLS
    elif tok == "L:NUM":
        set_bit(1)  # 数値ラベル
    elif tok == "L:LIST":
        set_bit(2)
    elif tok.startswith("L:SYM:"):
        set_bit(3)
        # さらに演算子に軽い識別を入れる（+,-,*,/,powなど）
        # 残余次元があればハッシュ1bit
        rem = d_model - (base+5)
        if rem > 0:
            h = stable_hash_u32(tok) % rem
            v[base+5+h] = 1.0
    elif tok.startswith("L:BOOL"):
        set_bit(4)
    elif tok == "L:NONE":
        set_bit(5)
    else:
        # 未知/その他 → 軽いハッシュ
        rem = d_model - (base+6)
        if rem > 0:
            h = stable_hash_u32(tok) % rem
            v[base+6+h] = 1.0
    return v

# =========================================================
# 4) データセット（固定/学習 埋め込みに対応）
# =========================================================
class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = [PAD, CLS]
        seen = set(uniq)
        for t in tokens:
            if t not in seen:
                uniq.append(t); seen.add(t)
        self.stoi = {t:i for i,t in enumerate(uniq)}
    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, 0) for t in toks]  # default PAD=0
    def __len__(self): return len(self.stoi)

class ExprDataset(Dataset):
    def __init__(self, samples: List[SexpSample], embed_mode: str = "fixed"):
        assert embed_mode in ("fixed","learned")
        self.samples = samples
        self.embed_mode = embed_mode
        # 語彙は learned のときだけ使う
        if embed_mode == "learned":
            all_tokens: List[str] = []
            for s in samples:
                toks, _ = tokenize_dyck(s.sexp)
                all_tokens.extend([CLS] + toks)
            self.vocab = Vocab(all_tokens)
        else:
            self.vocab = None  # 使わない

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
        toks, vals = tokenize_dyck(s.sexp)
        toks = [CLS] + toks
        vals = [None] + vals

        if self.embed_mode == "learned":
            ids  = self.vocab.encode(toks)
            item = {"tokens": toks, "input_ids": torch.tensor(ids, dtype=torch.long)}
        else:
            item = {"tokens": toks}  # 後で固定写像にする

        num_vals = [0.0 if v is None else ( (v - self.num_mean) / self.num_std ) for v in vals]
        num_mask = [0 if v is None else 1 for v in vals]
        y = (s.value - self.y_mean) / self.y_std

        item.update({
            "num_vals": torch.tensor(num_vals, dtype=torch.float),
            "num_mask": torch.tensor(num_mask, dtype=torch.long),
            "y": torch.tensor([y], dtype=torch.float),
        })
        return item

def collate_fixed(batch, d_model: int):
    # 最大長に合わせてパディング
    maxlen = max(len(b["tokens"]) for b in batch)
    B = len(batch)
    feats = torch.zeros(B, maxlen, d_model)
    num_vals = torch.zeros(B, maxlen)
    num_mask = torch.zeros(B, maxlen, dtype=torch.long)
    attn_mask = torch.zeros(B, maxlen, dtype=torch.long)
    y = torch.cat([b["y"] for b in batch], dim=0)

    for i,b in enumerate(batch):
        toks = b["tokens"]
        L = len(toks)
        for j,tok in enumerate(toks):
            feats[i,j] = token_to_fixed_vec(tok, d_model)
        # 数値
        nv = b["num_vals"]; nm = b["num_mask"]
        num_vals[i,:L] = nv[:L]
        num_mask[i,:L] = nm[:L]
        attn_mask[i,:L] = 1
    return {"input_feats": feats, "attn_mask": attn_mask,
            "num_vals": num_vals, "num_mask": num_mask, "y": y}

def collate_learned(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = 0  # <PAD>
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    num_vals  = torch.zeros(B, maxlen)
    num_mask  = torch.zeros(B, maxlen, dtype=torch.long)
    attn_mask = torch.zeros(B, maxlen, dtype=torch.long)
    y = torch.cat([b["y"] for b in batch], dim=0)

    for i,b in enumerate(batch):
        ids = b["input_ids"]; L = len(ids)
        input_ids[i,:L] = ids
        num_vals[i,:L] = b["num_vals"][:L]
        num_mask[i,:L] = b["num_mask"][:L]
        attn_mask[i,:L] = 1
    return {"input_ids": input_ids, "attn_mask": attn_mask,
            "num_vals": num_vals, "num_mask": num_mask, "y": y}

# =========================================================
# 5) Transformer 本体：固定/学習 どちらもOK
# =========================================================
class TransformerRegressor(nn.Module):
    def __init__(self, *, vocab_size: int | None, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512, max_len: int = 4096,
                 dropout: float = 0.1, embed_mode: str = "fixed"):
        super().__init__()
        assert embed_mode in ("fixed","learned")
        self.embed_mode = embed_mode
        if embed_mode == "learned":
            assert vocab_size is not None
            self.tok = nn.Embedding(vocab_size, d_model)
        else:
            self.tok = None  # 使わない

        self.pos = nn.Embedding(max_len, d_model)

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
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, *, input_ids=None, input_feats=None,
                num_vals: torch.Tensor, num_mask: torch.Tensor, attn_mask: torch.Tensor):
        B, L = (input_ids.shape if self.embed_mode=="learned" else input_feats.shape[:2])
        pos_ids = torch.arange(L, device=attn_mask.device).unsqueeze(0).expand(B, L)
        pos = self.pos(pos_ids)

        if self.embed_mode == "learned":
            x = self.tok(input_ids) + pos
        else:
            # input_feats はすでに d_model 次元
            x = input_feats + pos

        # 数値埋め込み（L:NUM にだけ入る）
        num_embed = self.num_proj(num_vals.unsqueeze(-1)) * num_mask.unsqueeze(-1).float()
        x = x + num_embed

        key_padding_mask = (attn_mask == 0)  # True=pad
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]
        return self.head(cls)

# =========================================================
# 6) 学習・評価ユーティリティ
# =========================================================
def split_data(samples: List[SexpSample], ratios=(0.8,0.1,0.1), seed=0):
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)
    n = len(samples); a = int(n*ratios[0]); b = int(n*ratios[1])
    return samples[:a], samples[a:a+b], samples[a+b:]

def compute_norms(train: List[SexpSample]):
    # 数値トークン統計（Dyck+ラベルの L:NUM から抽出）
    nums: List[float] = []
    ys:   List[float] = [s.value for s in train]
    for s in train:
        _, vals = tokenize_dyck(s.sexp)
        for v in vals:
            if v is not None and math.isfinite(v):
                nums.append(float(v))
    num_mean = float(sum(nums)/max(len(nums),1))_
