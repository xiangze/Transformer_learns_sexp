# transformer_train_eval.py
from __future__ import annotations
import re, ast, math, random, argparse, os
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import hy
from sexpdata import Symbol, dumps, loads

# =========================================================
# 0) ランダム数値S式データセット生成 (Hyで評価)
#    （前回の簡約版：数値式を生成→Hyで評価→(sexp_str, value)）
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
    from sexpdata import Symbol
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
        # (if cond then else) ただし then/else は数値式
        cond = _gen_comparison(depth - 1)
        then_e = _gen_numeric_expr(depth - 1)
        else_e = _gen_numeric_expr(depth - 1)
        return [Symbol("if"), cond, then_e, else_e]

def _gen_comparison(depth: int) -> Any:
    from sexpdata import Symbol
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

def make_sexp_eval_dataset(
    n_samples: int = 2000,
    *,
    max_depth: int = 4,
    seed: Optional[int] = 0,
    retries: int = 6,
    ensure_finite: bool = True,
) -> List[SexpSample]:
    if seed is not None:
        random.seed(seed)
    out: List[SexpSample] = []
    from sexpdata import dumps
    while len(out) < n_samples:
        for _ in range(retries):
            sexp_ast = _gen_numeric_expr(max_depth)
            sexp_str = dumps(sexp_ast)
            try:
                val = _eval_hy_expr_str(sexp_str)
                if isinstance(val, (int, float)):
                    f = float(val)
                    if ensure_finite and not math.isfinite(f):
                        continue
                    out.append(SexpSample(sexp=sexp_str, value=f))
                    break
            except Exception:
                continue
    return out

# =========================================================
# 1) S式 ↔ Dyck（括弧列＋ラベル列）  ※前回の往復のうち利用部
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
                 num_layers: int = 4, dim_ff: int = 1024, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, input_ids: torch.Tensor, num_vals: torch.Tensor,
                num_mask: torch.Tensor, attn_mask: torch.Tensor):
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok(input_ids) + self.pos(pos_ids)

        # numeric embedding（非数値位置は0）
        num_embed = self.num_proj(num_vals.unsqueeze(-1))
        num_embed = num_embed * num_mask.unsqueeze(-1).float()
        x = x + num_embed

        key_padding_mask = (attn_mask == 0)  # True=padding
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]  # 先頭が <CLS>
        yhat = self.head(cls)  # (B,1)
        return yhat

# =========================================================
# 6) 学習・評価ループ
# =========================================================

def split_data(samples: List[SexpSample], ratios=(0.8, 0.1, 0.1), seed=0):
    random.Random(seed).shuffle(samples)
    n = len(samples)
    n_train = int(n*ratios[0])
    n_val   = int(n*ratios[1])
    train = samples[:n_train]
    val   = samples[n_train:n_train+n_val]
    test  = samples[n_train+n_val:]
    return train, val, test

def compute_norms(train: List[SexpSample], mode: str):
    # 数値トークンの mean/std, 目的変数の mean/std
    nums: List[float] = []
    ys:   List[float] = [s.value for s in train]
    for s in train:
        toks, vals = (tokenize_sexp(s.sexp) if mode=="sexp" else tokenize_dyck(s.sexp))
        for v in vals:
            if v is not None and math.isfinite(v):
                nums.append(float(v))
    num_mean = float(sum(nums)/max(len(nums),1)) if nums else 0.0
    num_std  = float((sum((x-num_mean)**2 for x in nums)/max(len(nums),1))**0.5) if nums else 1.0
    y_mean   = float(sum(ys)/len(ys))
    y_std    = float((sum((y-y_mean)**2 for y in ys)/len(ys))**0.5)
    return num_mean, max(num_std,1e-6), y_mean, max(y_std,1e-6)

@torch.no_grad()
def eval_loop(model, loader, device, y_mean, y_std):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    preds, trues = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        num_vals  = batch["num_vals"].to(device)
        num_mask  = batch["num_mask"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        y         = batch["y"].to(device)
        yhat = model(input_ids, num_vals, num_mask, attn_mask)
        # 逆正規化
        yhat_den = yhat * y_std + y_mean
        y_den    = y * y_std + y_mean
        diff = (yhat_den - y_den).squeeze(-1)
        mse += torch.mean(diff**2).item()*len(y)
        mae += torch.mean(torch.abs(diff)).item()*len(y)
        n += len(y)
        preds += yhat_den.squeeze(-1).cpu().tolist()
        trues += y_den.squeeze(-1).cpu().tolist()
    mse /= max(n,1); mae /= max(n,1)
    # R^2
    import numpy as np
    y_arr = np.array(trues); ybar = y_arr.mean()
    ss_tot = ((y_arr - ybar)**2).sum()
    ss_res = ((y_arr - np.array(preds))**2).sum()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return mse, mae, r2, preds, trues

def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # ---- データ生成
    samples = make_sexp_eval_dataset(
        n_samples=args.n_samples,
        max_depth=args.max_depth,
        seed=args.seed,
        retries=8,
        ensure_finite=True,
    )

    train_s, val_s, test_s = split_data(samples, seed=args.seed)
    ds_train = ExprDataset(train_s, mode=args.mode)
    ds_val   = ExprDataset(val_s,   mode=args.mode)
    ds_test  = ExprDataset(test_s,  mode=args.mode)

    # 正規化パラメータ（train から算出 → 全splitに適用）
    num_mean, num_std, y_mean, y_std = compute_norms(train_s, args.mode)
    for ds in (ds_train, ds_val, ds_test):
        ds.set_norms(num_mean, num_std, y_mean, y_std)

    # DataLoader
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---- モデル
    model = TransformerRegressor(
        vocab_size=len(ds_train.vocab),
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        max_len=args.max_len, dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.SmoothL1Loss()  # MSEでもOK

    # ---- 学習
    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0; count = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            num_vals  = batch["num_vals"].to(device)
            num_mask  = batch["num_mask"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            y         = batch["y"].to(device)

            yhat = model(input_ids, num_vals, num_mask, attn_mask)
            loss = loss_fn(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()

            running += loss.item() * len(y)
            count   += len(y)

        train_loss = running / max(count,1)
        val_mse, val_mae, val_r2, _, _ = eval_loop(model, val_loader, device, y_mean, y_std)

        print(f"[{epoch:03d}] train_loss={train_loss:.4f}  "
              f"val_MSE={val_mse:.4f}  val_MAE={val_mae:.4f}  val_R2={val_r2:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            if args.ckpt:
                torch.save({
                    "model": model.state_dict(),
                    "vocab": ds_train.vocab.stoi,
                    "args": vars(args),
                    "norms": (num_mean, num_std, y_mean, y_std),
                    "mode": args.mode,
                }, args.ckpt)

    # ---- テスト評価
    test_mse, test_mae, test_r2, preds, trues = eval_loop(model, test_loader, device, y_mean, y_std)
    print(f"[TEST]  MSE={test_mse:.4f}  MAE={test_mae:.4f}  R2={test_r2:.4f}")

    # サンプルを数件表示
    print("\nExamples (prediction -> truth):")
    for i in range(min(10, len(preds))):
        print(f"  {preds[i]:.4f}  ->  {trues[i]:.4f}")

# =========================================================
# 7) エントリポイント
# =========================================================

def build_argparser():
    p = argparse.ArgumentParser(description="Train Transformer on S-expr evaluation")
    p.add_argument("--mode", choices=["sexp","dyck"], default="sexp",
                   help="Input representation")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=2048)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ckpt", type=str, default="")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_main(args)
