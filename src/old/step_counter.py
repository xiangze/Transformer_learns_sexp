# ==== Step-counting extension for evallisp.py ====
# 使い方:
#   from evallisp import eval_with_steps
#   val, steps = eval_with_steps("(+ 1 2)")
# CLI:
#   python -m evallisp --steps "(+ 1 2)"   # 値とステップ数を表示
#
# 「1 ステップ」= プリミティブ演算 ( + - * / % == != > < >= <= list car cdr など )
#                 または特殊形式の処理 1 回（フォールバック時の近似）

from functools import wraps
import sys
import types

STEP_COUNT = 0

def _step(n: int = 1):
    global STEP_COUNT
    STEP_COUNT += n

def reset_step_counter():
    global STEP_COUNT
    STEP_COUNT = 0

class _CountingPrimitive:
    def __init__(self, fn, name=None):
        self.fn = fn
        self.__name__ = name or getattr(fn, "__name__", "prim")
    def __call__(self, *args, **kwargs):
        _step(1)
        return self.fn(*args, **kwargs)

def _wrap_callables_in_env(env_like):
    """辞書/Env風オブジェクトの callables を全部ラップしてステップ加算。"""
    try:
        # dict 互換
        keys = list(env_like.keys())
        new_env = env_like.copy() if hasattr(env_like, "copy") else dict(env_like)
        for k in keys:
            v = env_like[k]
            if callable(v):
                new_env[k] = _CountingPrimitive(v, name=str(k))
        return new_env
    except Exception:
        # 属性アクセス型の Env の場合 (最低限の対応)
        new_env = {}
        for k in dir(env_like):
            if k.startswith("_"): 
                continue
            v = getattr(env_like, k)
            if callable(v):
                new_env[k] = _CountingPrimitive(v, name=str(k))
        return new_env

def _find_evaluator():
    """このモジュール内の評価器候補を探す。優先順: eval_with_env, eval_lisp, evaluate, eval"""
    cand_names = ("eval_with_env", "eval_lisp", "evaluate", "eval")
    for nm in cand_names:
        f = globals().get(nm)
        if callable(f):
            return f
    return None

def _find_parser():
    """パーサ候補 (expr -> AST)。なくても OK。"""
    cand_names = ("parse", "read", "read_from_tokens")
    for nm in cand_names:
        f = globals().get(nm)
        if callable(f):
            return f
    return None

# ---- フォールバック: 最小限の評価器 (数値・比較・list/car/cdr のみ) ----
def _tokenize(s: str):
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def _atom(tok: str):
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok

def _read_from_tokens(ts):
    if not ts: 
        raise SyntaxError("unexpected EOF")
    t = ts.pop(0)
    if t == "(":
        L = []
        while ts[0] != ")":
            L.append(_read_from_tokens(ts))
        ts.pop(0)
        return L
    elif t == ")":
        raise SyntaxError("unexpected )")
    else:
        return _atom(t)

def _parse_simple(expr):
    return _read_from_tokens(_tokenize(expr))

def _apply_prim(op, args):
    _step(1)  # プリミティブ適用 1 ステップ
    if op == "+":  return args[0] + args[1]
    if op == "-":  return args[0] - args[1]
    if op == "*":  return args[0] * args[1]
    if op == "/":  return args[0] / args[1]
    if op == "%":  return args[0] % args[1]
    if op == "==": return args[0] == args[1]
    if op == "!=": return args[0] != args[1]
    if op == ">":  return args[0] >  args[1]
    if op == "<":  return args[0] <  args[1]
    if op == ">=": return args[0] >= args[1]
    if op == "<=": return args[0] <= args[1]
    if op == "list": return list(args)
    if op == "car":  return args[0][0]
    if op == "cdr":  return args[0][1:]
    raise ValueError(f"unknown op: {op}")

def _eval_fallback(ast, env):
    # 近似: ノード到達ごとに 1 ステップ
    _step(1)
    if isinstance(ast, (int, float, bool, str)):
        if isinstance(ast, str) and ast in env:
            return env[ast]
        return ast
    if not ast: 
        return ast
    op = _eval_fallback(ast[0], env)
    args = [ _eval_fallback(a, env) for a in ast[1:] ]
    if isinstance(op, str):
        return _apply_prim(op, args)
    # ラムダやクロージャ形は非対応（用途上、原始関数と記号のみで十分なはず）
    raise TypeError("unsupported application in fallback")

# ---- 公開 API ----
def eval_with_steps(expr, env=None):
    """
    expr を評価して (value, steps) を返す。
    - まずモジュール内の評価器 (eval_lisp/evaluate/eval など) を探す
      - 環境が辞書型なら callables をラップして「プリミティブ適用=1 ステップ」
      - 呼び出しシグネチャは (expr, env) / (expr) / (AST, env) / (AST) を順に試す
    - 見つからない/包めない場合は、最小限のフォールバック評価器で近似カウント
    """
    reset_step_counter()

    # 1) 既存の評価器があればそれを使う
    _eval = _find_evaluator()
    if _eval is not None:
        # 既存 env を見つける（なければ None のまま）
        base_env = env
        if base_env is None:
            for name in ("GLOBAL_ENV", "global_env", "ENV", "env", "standard_env", "PRIMS"):
                if name in globals():
                    base_env = globals()[name]
                    break
        counting_env = _wrap_callables_in_env(base_env) if base_env is not None else None

        # 既存パーサ
        _parse = _find_parser()

        # 呼び出し形を総当り
        tried = []
        def _try_call(*args, **kwargs):
            tried.append((args, kwargs))
            return _eval(*args, **kwargs)

        try:
            if counting_env is not None:
                return _try_call(expr, counting_env), STEP_COUNT
        except TypeError:
            pass
        try:
            return _try_call(expr), STEP_COUNT
        except TypeError:
            pass
        if _parse is not None:
            ast = _parse(expr) if isinstance(expr, str) else expr
            try:
                if counting_env is not None:
                    return _try_call(ast, counting_env), STEP_COUNT
            except TypeError:
                pass
            try:
                return _try_call(ast), STEP_COUNT
            except TypeError:
                pass
        # うまく包めない場合はフォールバックへ

    # 2) フォールバック評価
    ast = _parse_simple(expr) if isinstance(expr, str) else expr
    value = _eval_fallback(ast, env or {})
    return value, STEP_COUNT

# ---- 簡易 CLI: python -m evallisp --steps "(+ 1 2)" ----
def _cli_steps(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "--steps":
        expr = " ".join(argv[1:])
        val, steps = eval_with_steps(expr)
        print(val)
        print(f"[steps]={steps}")
        return True
    return False

# 既存の __main__ がある場合の衝突を避け、--steps の時だけここで処理
if __name__ == "__main__":
    if _cli_steps():
        sys.exit(0)

