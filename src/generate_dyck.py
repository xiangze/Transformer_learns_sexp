from __future__ import annotations
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import hy  # HyをPythonから評価に使う
from sexpdata import Symbol, dumps  # S式の構築/文字列化

# ----------------------------
# ランダムS式ジェネレータ（数値式に特化）
# ----------------------------

_NUM_UNARY_FUNCS = ["abs", "round"]      # 1引数
_NUM_BINARY_OPS  = ["+", "-", "*", "/"]  # 2引数（/ は浮動小数になる）
_NUM_NARY_FUNCS  = ["max", "min", "sum"] # 可変長
_POW_FUNC        = "pow"                 # (pow x k)

_CMP_OPS         = ["<", "<=", ">", ">=", "=", "!="]  # 条件式用
# Hyでは '=' が等価比較（Pythonの==）に対応

def _rand_number(int_ratio: float = 0.6) -> float | int:
    """整数/小数をランダム生成（過大化を避けるために範囲を絞る）"""
    if random.random() < int_ratio:
        return random.randint(-9, 9)
    # 小数は丸めておく
    return round(random.uniform(-9.0, 9.0), 2)

def _rand_small_int(low: int = -4, high: int = 4) -> int:
    x = 0
    while x == 0:
        x = random.randint(low, high)
    return x

def _gen_numeric_expr(depth: int) -> Any:
    """
    数値を返すHy式のS式(AST相当; sexpdataのSymbol/数/リスト)を生成。
    depthが0または確率的に打ち切りなら原子（数値）。
    """
    if depth <= 0 or random.random() < 0.35:
        return _rand_number()

    choice = random.random()
    if choice < 0.35:
        # 二項演算 (+ - * /)
        op = random.choice(_NUM_BINARY_OPS)
        a = _gen_numeric_expr(depth - 1)
        b = _gen_numeric_expr(depth - 1)
        return [Symbol(op), a, b]

    elif choice < 0.55:
        # 単項関数 (abs x) / (round x)
        f = random.choice(_NUM_UNARY_FUNCS)
        x = _gen_numeric_expr(depth - 1)
        return [Symbol(f), x]

    elif choice < 0.7:
        # (pow base k) ただし指数は小さめの整数
        base = _gen_numeric_expr(depth - 1)
        k = _rand_small_int()
        return [Symbol(_POW_FUNC), base, k]

    elif choice < 0.85:
        # 可変長関数 (sum (list ...)) / (max (list ...)) / (min (list ...))
        f = random.choice(_NUM_NARY_FUNCS)
        n_items = random.randint(2, 5)
        items = [ _gen_numeric_expr(depth - 1) for _ in range(n_items) ]
        return [Symbol(f), [Symbol("list"), *items]]

    else:
        # if式: (if 条件 then-expr else-expr) すべて数値にする
        cond = _gen_comparison(depth - 1)
        then_e = _gen_numeric_expr(depth - 1)
        else_e = _gen_numeric_expr(depth - 1)
        return [Symbol("if"), cond, then_e, else_e]

def _gen_comparison(depth: int) -> Any:
    op = random.choice(_CMP_OPS)
    a = _gen_numeric_expr(depth - 1)
    b = _gen_numeric_expr(depth - 1)
    return [Symbol(op), a, b]

# ----------------------------
# Hyで評価
# ----------------------------
def eval_hy_expr_str(expr_str: str) -> Any:
    # hy.read_str は文字列をHyのモデル（抽象構文オブジェクト）に変換
    # hy.eval はそのモデルを評価してPython値を返す
    model = hy.read_str(expr_str)
    return hy.eval(model)  # 必要なら module=… を渡して環境を分けられます

# ----------------------------
# データセット作成
# ----------------------------

@dataclass
class SexpSample:
    sexp: str
    value: Any

def make_sexp_eval_dataset(
    n_samples: int = 200,
    *,
    max_depth: int = 4,
    seed: Optional[int] = None,
    retries: int = 8,
    ensure_finite: bool = True,
) -> List[SexpSample]:
    """
    ランダムS式を生成しHyで評価した(式文字列, 値)のペアを n_samples 個つくる。

    - max_depth: 生成木の最大深さ
    - retries: 評価エラー(ゼロ除算・オーバーフロー等)時の再試行回数
    - ensure_finite: 数値結果がinf/NaNなら捨てて再試行
    """
    if seed is not None:
        random.seed(seed)

    out: List[SexpSample] = []
    while len(out) < n_samples:
        ok = False
        for _ in range(retries):
            sexp_ast = _gen_numeric_expr(max_depth)
            sexp_str = dumps(sexp_ast)

            try:
                val = eval_hy_expr_str(sexp_str)

                # 数値なら有限性チェック
                if ensure_finite and isinstance(val, (int, float)):
                    if not math.isfinite(float(val)):
                        continue

                out.append(SexpSample(sexp=sexp_str, value=val))
                ok = True
                break

            except Exception:
                # 評価に失敗（ゼロ除算・オーバーフロー・型エラーなど）→作り直し
                continue

        if not ok:
            # 何度やっても失敗する場合は深さを浅くして救済
            try:
                sexp_ast = _gen_numeric_expr(max_depth=2)
                sexp_str =_

"""
random Dick_k generator
k: 括弧の種類数
length: 文字列長
N: 文字列の数

"""
def generate_dick(k=3,length=10,N=1000,eps=0.):
    ls=[]
    for n in range(N):
        for l in range(length):
            d=
        ls.append(d)
    return ls

def generate_S(k=3,length=10,N=1000,eps=0.):
    pass
