# Python 3.9+ で動作

from math import fabs as _fabs, pow as _pow
from operator import add, sub, mul, truediv, lt, le, gt, ge, eq, ne
from typing import Any, Dict, List, Tuple, Union

SExp = Union[list, tuple, str, int, float, bool] #Any?

# ---------- predicates ----------
def is_symbol(x: Any) -> bool:    return isinstance(x, str)
def is_seq(x: Any) -> bool:    return isinstance(x, (list, tuple))
def is_numeric(x: Any) -> bool:    return isinstance(x, (int, float))
def sym_eq(a: str, b: str) -> bool:    return a == b

def free_of(sym: str, form: SExp) -> bool:
    """form に sym が含まれなければ True"""
    if is_symbol(form):
        return not sym_eq(form, sym)
    if is_seq(form):
        return all(free_of(sym, e) for e in form)
    return True

# ---------- safe ops (whitelist) ----------
def _safe_abs(a):  return abs(float(a))
def _safe_pow(a,b): return _pow(float(a), float(b))
def _safe_list(*xs): return list(xs)
def _safe_max(*xs):
    # (max (list ...)) / (max a b c) の両対応
    if len(xs) == 1 and isinstance(xs[0], list):
        xs = tuple(xs[0])
    return max(xs)

SAFE_OP: Dict[str, Any] = {
    '+': add, '-': sub, '*': mul, #'/': truediv,
    '=': eq, '!=': ne, '<': lt, '<=': le, '>': gt, '>=': ge,
    'abs': _safe_abs, 'pow': _safe_pow, 'list': _safe_list, 'max': _safe_max,
}

# ---------- light algebraic simplification ----------
def simplify(form: SExp) -> SExp:
    """x を評価せずに安全な恒等変形だけ行う"""
    if not is_seq(form):
        return form

    op, *args = form
    args = [simplify(a) for a in args]

    # (* 0 anything) -> 0
    if op == '*' and any(a == 0 for a in args):
        return 0

    # (if true a b) / (if false a b)
    if op == 'if' and args and isinstance(args[0], bool):
        return simplify(args[1] if args[0] else args[2])

    # (if cond t t) -> t
    if op == 'if' and len(args) == 3 and args[1] == args[2]:
        return simplify(args[1])

    # (max (list (abs x) ...)) の数値だけ集約
    if op == 'max' and len(args) == 1 and is_seq(args[0]) and args[0] and args[0][0] == 'list':
        ls = args[0][1:]  # list の中身
        if not ls:
            return float('-inf')
        # 先頭をそのまま残す（例: (abs x)）
        head, *tail = ls
        nums = [t for t in tail if is_numeric(t)]
        syms = [t for t in tail if not is_numeric(t)]
        if nums:
            return ['max', head, max(nums), *syms]
        return form

    return [op, *args]

# ---------- partial evaluator ----------
def peval_except(sym: str, form: SExp) -> SExp:
    """sym を自由変数として残しつつ部分評価"""
    def go(e: SExp, env: Dict[str, Any]) -> SExp:
        # 基本型
        if is_numeric(e) or isinstance(e, bool):
            return e
        if is_symbol(e):
            return env[e] if e in env else e

        # S式
        if is_seq(e):
            op, *args = e

            # do
            if op == 'do':
                last = None
                for a in args:
                    last = go(a, env)
                return last

            # if
            if op == 'if':
                c = go(args[0], env)
                if isinstance(c, bool):
                    return go(args[1] if c else args[2], env)
                return ['if', c, go(args[1], env), go(args[2], env)]

            # (fn [t] body) -> closure
            if op == 'fn':
                param_list = args[0]  # ['t', ...] を想定（単引数）
                body = args[1]
                return {'closure': {'param': param_list[0], 'body': body, 'env': dict(env)}}

            # それ以外：関数適用 or 通常の演算
            try:
                opv = go(op, env)
            except:
                return e

            # λ適用
            if isinstance(opv, dict) and 'closure' in opv:
                if free_of(sym, e):
                    cl = opv['closure']
                    argv = [go(a, env) for a in args]
                    new_env = dict(cl['env'])
                    new_env[cl['param']] = argv[0]  # 単引数
                    return go(cl['body'], new_env)
                # x を含むなら評価せず子だけ再帰
                return [opv, *[go(a, env) for a in args]]

            # ホワイトリスト演算：式全体が x を含まなければ評価
            if is_symbol(op) and op in SAFE_OP and free_of(sym, e):
                return SAFE_OP[op](*[go(a, env) for a in args])

            # それ以外：子を再帰し、軽く簡約
            return simplify([op, *[go(a, env) for a in args]])

        # その他はそのまま
        return e

    return go(simplify(form), {})

# ---------- demo ----------
if __name__ == "__main__":
    # ((fn [t] (if (< (* 0 (abs t))
    #                (max (list (abs x) ((fn [z] -5) 0.44) ((fn [b] 1) t) t)))
    #            (* (if (= 6 -6) -8 -5.93)
    #               (if (!= v t) t t))
    #            (pow (if (= 2 2) t 3) 4))) 8)
    expr: SExp = [
        'do',
        [
            ['fn', ['t'],
                ['if',
                    ['<',
                        ['*', 0, ['abs', 't']],
                        ['max', ['list',
                                 ['abs', 'x'],
                                 [[ 'fn', ['z'], -5], 0.44],
                                 [[ 'fn', ['b'], 1], 't'],
                                 't']]],

                    ['*',
                        ['if', ['=', 6, -6], -8, -5.93],
                        ['if', ['!=', 'v', 't'], 't', 't']],

                    ['pow', ['if', ['=', 2, 2], 't', 3], 4]
                ]
            ],
            8
        ]
    ]
    out = peval_except('x', expr)
    print(out)           # => -47.44
