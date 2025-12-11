import argparse
import json
import math
import os
import random
import re
import sys
import importlib.util
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

###############################################################################
# S-Expression generator
###############################################################################

BIN_ARITH = ["+", "-", "*", "/", "%"]
BIN_CMP   = ["==", "!=", ">", "<", ">=", "<="]
UNARY_LIST = ["car", "cdr"]

Symbol = str  # for clarity

def _choice(seq):
    return random.choice(seq)

def _rand_int():
    # 小さめの定数（0 も可。ただし /, % の右辺には使わない）
    return random.randint(-9, 9)

def _var_name(i: int) -> str:
    return f"x{i}"

class GenConfig:
    def __init__(self,
                 num_free_vars: int = 3,
                 max_depth: int = 3,
                 share_prev_expr_prob: float = 0.35) -> None:
        self.num_free_vars = num_free_vars
        self.max_depth = max_depth
        # 同一要素内で「前の式をサブ式として組み込む」確率（依存関係を強める）
        self.share_prev_expr_prob = share_prev_expr_prob

class SExprGenerator:
    """
    - 指定の演算子のみを使って S 式を生成
    - `car`/`cdr` 用に (list ...) を最小限だけ使用（評価器依存を避けるため）
    - later_expr に earlier_expr をサブ式として埋め込み、依存関係を作る
    """
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.vars: List[str] = [_var_name(i) for i in range(cfg.num_free_vars)]

    def generate_groups(self,
                        num_groups: int,
                        min_exprs: int,
                        max_exprs: int,
                        seed: Optional[int] = None) -> List[List[str]]:
        if seed is not None:
            random.seed(seed)
        all_groups: List[List[str]] = []
        for _ in range(num_groups):
            k = random.randint(min_exprs, max_exprs)
            group: List[str] = []
            prev_exprs: List[str] = []
            for j in range(k):
                # ときどき前の式をサブ式として利用して依存関係を作る
                use_prev = (j > 0) and (random.random() < self.cfg.share_prev_expr_prob)
                prev_fragment = _choice(prev_exprs) if use_prev else None
                expr = self._gen_expr(want="num", depth=self.cfg.max_depth, prev_fragment=prev_fragment)
                group.append(expr)
                prev_exprs.append(expr)
            all_groups.append(group)
        return all_groups

    # want ∈ {"num","bool","list"}
    def _gen_expr(self, want: str, depth: int, prev_fragment: Optional[str]) -> str:
        if depth <= 0:
            return self._gen_atom(want)

        # たまに前の式をそのままサブ式に使う（型はざっくり num 扱い）
        if prev_fragment and want != "list" and random.random() < 0.5:
            # 既存式に軽い算術/比較をかぶせる
            if want == "num":
                op = _choice(BIN_ARITH)
                if op in ("/", "%"):
                    rhs = str(random.randint(1, 9))  # 0 を回避
                else:
                    rhs = self._gen_atom("num")
                return f"({op} {prev_fragment} {rhs})"
            if want == "bool":
                op = _choice(BIN_CMP)
                rhs = self._gen_atom("num")
                return f"({op} {prev_fragment} {rhs})"

        if want == "num":
            # num を返す式：算術 or car(list ...) or car(cdr(list ...))
            choice = random.random()
            if choice < 0.6:
                op = _choice(BIN_ARITH)
                lhs = self._gen_expr("num", depth-1, prev_fragment)
                if op in ("/", "%"):
                    # 右辺は非ゼロの定数に限定
                    rhs = str(random.randint(1, 9))
                else:
                    rhs = self._gen_expr("num", depth-1, prev_fragment)
                return f"({op} {lhs} {rhs})"
            else:
                # car の結果を num 扱い
                lst = self._gen_list_of_nums(depth-1, prev_fragment)
                return f"(car {lst})"

        elif want == "bool":
            op = _choice(BIN_CMP)
            lhs = self._gen_expr("num", depth-1, prev_fragment)
            rhs = self._gen_expr("num", depth-1, prev_fragment)
            return f"({op} {lhs} {rhs})"

        else:  # want == "list"
            # cdr(list ...) や list 自体
            if random.random() < 0.5:
                return f"(cdr {self._gen_list_of_nums(depth-1, prev_fragment)})"
            else:
                return self._gen_list_of_nums(depth-1, prev_fragment)

    def _gen_atom(self, want: str) -> str:
        if want == "num":
            if self.vars and random.random() < 0.6:
                return _choice(self.vars)
            return str(_rand_int())
        elif want == "bool":
            # 比較で bool を作る方が自然なので、ここでは簡単に固定
            # 真偽値リテラルを避ける（評価器の仕様差異を避ける）
            lhs = self._gen_atom("num")
            rhs = self._gen_atom("num")
            return f"(== {lhs} {rhs})"
        else:  # want == "list"
            return self._gen_list_of_nums(0, None)

    def _gen_list_of_nums(self, depth: int, prev_fragment: Optional[str]) -> str:
        # (list e1 e2 e3 ...)
        m = random.randint(2, 4)
        items = [ self._gen_expr("num", max(depth-1,0), prev_fragment) for _ in range(m) ]
        return f"(list {' '.join(items)})"

###############################################################################
# Simple s-expression parser & evaluator (fallback with step counting)
###############################################################################

Token = str
AST = Union[Symbol, int, List[Any]]

def tokenize(s: str) -> List[Token]:
    # 最低限のトークナイザ（数値・記号・括弧）
    s = s.replace("(", " ( ").replace(")", " ) ")
    return [t for t in s.split() if t]

def atom(token: Token) -> AST:
    # int に落ちるものだけ int。他はシンボル。
    try:
        return int(token)
    except ValueError:
        return token

def read_from_tokens(tokens: List[Token]) -> AST:
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF")
    t = tokens.pop(0)
    if t == "(":
        L: List[AST] = []
        while tokens[0] != ")":
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # ')'
        return L
    elif t == ")":
        raise SyntaxError("unexpected )")
    else:
        return atom(t)

def parse(sexp: str) -> AST:
    return read_from_tokens(tokenize(sexp))

class StepCounter:
    def __init__(self) -> None:
        self.steps = 0
    def tick(self, n: int = 1):
        self.steps += n

def _apply_prim(op: str, args: List[Any]) -> Any:
    if op in ("+", "-", "*", "/", "%"):
        a, b = args
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return int(a / b)  # ここでは整数化
        if op == "%": return a % b
    if op in ("==","!=",">","<",">=","<="):
        a, b = args
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == ">":  return a >  b
        if op == "<":  return a <  b
        if op == ">=": return a >= b
        if op == "<=": return a <= b
    if op == "list":
        return list(args)
    if op == "car":
        (lst,) = args
        return lst[0]
    if op == "cdr":
        (lst,) = args
        return lst[1:]
    raise ValueError(f"unknown op: {op}")

def eval_fallback(ast: AST, env: Dict[str,int], sc: StepCounter) -> Any:
    """
    許可演算子のみ対応。各ノード処理で 1 ステップ加算（ざっくりな簡約ステップ近似）。
    """
    sc.tick(1)
    # 原子
    if isinstance(ast, int): return ast
    if isinstance(ast, str):
        if ast in env: return env[ast]
        # 'true' 'false' など未使用なのでそのまま返す
        return ast
    # リスト（関数適用）
    if not ast:
        return ast
    op = ast[0]
    args = [eval_fallback(a, env, sc) for a in ast[1:]]
    return _apply_prim(op, args)

###############################################################################
# evallist.py integration (best-effort)
###############################################################################

def load_evallist_module(path: str):
    spec = importlib.util.spec_from_file_location("evallist", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evallist from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def try_eval_with_evallist(evallist_mod,
                           sexp: str,
                           env: Dict[str,int]) -> Tuple[Optional[Any], Optional[int], Optional[str]]:
    """
    可能な関数名を総当りで試し、(value, steps) を取りに行く。
    未対応なら (value, None) か (None, None) を返す。
    """
    candidates = [
        # (func_name, pass_env?), ordered by likelihood
        ("eval_with_steps", True),
        ("eval_steps", True),
        ("eval_lisp_steps", True),
        ("evaluate_steps", True),
        ("eval_lisp", True),
        ("evaluate", True),
        ("eval", True),     # たいてい (expr, env) or (expr)
        ("run", True),
    ]

    last_err = None
    for fname, pass_env in candidates:
        f = getattr(evallist_mod, fname, None)
        if f is None:
            continue
        try:
            # いくつかの引数形を試す
            for args in ( (sexp, env), (sexp, ), (parse(sexp), env), (parse(sexp),) ):
                try:
                    ret = f(*args)  # 返り値の形は実装依存
                    # 返り値候補の解釈
                    if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], int):
                        return ret[0], int(ret[1]), fname
                    # ステップがグローバルにあるケース
                    steps = None
                    for attr in ("STEP_COUNT", "STEPS", "step_count", "steps"):
                        if hasattr(evallist_mod, attr) and isinstance(getattr(evallist_mod, attr), int):
                            steps = int(getattr(evallist_mod, attr))
                            break
                    return ret, steps, fname
                except TypeError:
                    continue  # 形が合わない
        except Exception as e:
            last_err = str(e)
            continue
    return None, None, last_err

###############################################################################
# Utilities
###############################################################################

def sample_env(num_free_vars: int, lo: int = -5, hi: int = 5) -> Dict[str,int]:
    env = {}
    for i in range(num_free_vars):
        # /,% で 0 除算を引かないよう、0 も許すが右辺は定数にしているので OK
        env[_var_name(i)] = random.randint(lo, hi)
    return env

def evaluate_group_with_steps(group: List[str],
                              env: Dict[str,int],
                              evallist_path: Optional[str]) -> List[Dict[str, Any]]:
    rows = []
    evm = None
    if evallist_path:
        try:
            evm = load_evallist_module(evallist_path)
        except Exception as e:
            print(f"[warn] failed to load evallist: {e}", file=sys.stderr)
            evm = None

    for expr in group:
        val, steps, backend = None, None, None
        if evm is not None:
            val, steps, backend = try_eval_with_evallist(evm, expr, env)
        if steps is None:
            # フォールバックで評価 + ステップ近似
            try:
                sc = StepCounter()
                val = eval_fallback(parse(expr), env, sc)
                steps = sc.steps
                backend = backend or "fallback"
            except Exception as e:
                val = f"ERROR: {e}"
                steps = None
                backend = backend or "fallback"
        rows.append({"expr": expr, "value": val, "steps": steps, "backend": backend})
    return rows

###############################################################################
# CLI
###############################################################################

def main():
    ap = argparse.ArgumentParser(description="Random S-expression groups generator + (optional) step tester via evallist.py")
    ap.add_argument("--groups", type=int, default=3, help="外側リストの要素数（=グループ数）")
    ap.add_argument("--min-exprs", type=int, default=2, help="1グループ内の最小 S 式数")
    ap.add_argument("--max-exprs", type=int, default=4, help="1グループ内の最大 S 式数")
    ap.add_argument("--free-vars", type=int, default=3, help="自由変数の数")
    ap.add_argument("--max-depth", type=int, default=3, help="S 式の最大入れ子深さ")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード")
    ap.add_argument("--evallist-path", type=str, default=None, help="Transformer_learns_sexp/src/evallist.py のパス")
    ap.add_argument("--dump-json", type=str, default=None, help="生成物を JSON で保存するファイルパス")
    args = ap.parse_args()

    cfg = GenConfig(num_free_vars=args.free_vars, max_depth=args.max_depth)
    gen = SExprGenerator(cfg)
    groups = gen.generate_groups(args.groups, args.min_exprs, args.max_exprs, seed=args.seed)

    # 画面に例を表示
    print("# Generated S-expression groups (list of list of strings):")
    for gi, g in enumerate(groups):
        print(f"- group[{gi}] (|S|={len(g)}):")
        for e in g:
            print("   ", e)

    # テスト: 各グループについてランダム環境で評価 & ステップ
    print("\n# Evaluation test (random env per group):")
    all_out = []
    for gi, g in enumerate(groups):
        env = sample_env(args.free_vars)
        rows = evaluate_group_with_steps(g, env, args.evallist_path)
        print(f"## group[{gi}] env={env}")
        for r in rows:
            print(f"   expr: {r['expr']}")
            print(f"   -> value: {r['value']}    steps: {r['steps']}    backend: {r['backend']}")
        all_out.append({"env": env, "rows": rows})

    if args.dump_json:
        payload = {"groups": groups, "eval": all_out}
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON to: {args.dump_json}")

if __name__ == "__main__":
    main()
