"""
Random higher‑order S‑expression generator + evaluator (partial evaluation)
Rewritten in **pure Python** while using the **hy** library's data types (e.g., Symbol).

Features
--------
- Randomly generate S‑expressions involving higher‑order functions:
  (fn [...]), application, compose, partial, map/filter/reduce, if, + - * /, comparisons.
- Evaluate to a concrete value when possible (closed terms / sufficient info).
- If free variables remain, return a **simplified** S‑expression (partial evaluation).

Usage
-----
    python random_hof_sexpr.py --n 10 --max-depth 4 --seed 0
    python random_hof_sexpr.py --n 5 --closed

Dependencies
------------
    pip install hy

Note
----
This script does **not** execute Hy code; it builds S‑expressions using
`hy.models.Symbol` and evaluates them with a Python interpreter we provide here.
"""

from __future__ import annotations
import argparse
import math
import operator as op
import random
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Tuple, Union

from hy.models import Symbol  # only data structure usage

SExpr = Union[int, float, bool, Symbol, List["SExpr"]]
Env = Dict[Symbol, Any]

# -----------------------------
# Utility predicates
# -----------------------------

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def is_symbol(x: Any) -> bool:
    return isinstance(x, Symbol)


def literal_value(x: Any) -> bool:
    if is_number(x) or is_bool(x):
        return True
    if isinstance(x, list):
        return all(literal_value(e) for e in x)
    return False


def S(name: str) -> Symbol:
    return Symbol(name)

FREE_SYMS = [S("x"), S("y"), S("z"), S("t")]

# -----------------------------
# Closures
# -----------------------------

@dataclass
class Closure:
    params: List[Symbol]
    body: SExpr
    env: Env


def is_closure(v: Any) -> bool:
    return isinstance(v, Closure)

# -----------------------------
# Pretty printer
# -----------------------------

def pprint_sexpr(form: SExpr) -> str:
    if is_symbol(form):
        return form.name
    if is_number(form) or is_bool(form):
        return str(form)
    if isinstance(form, list):
        return "(" + " ".join(pprint_sexpr(x) for x in form) + ")"
    return repr(form)

# -----------------------------
# Environment helpers
# -----------------------------

def env_extend(env: Env, keys: List[Symbol], vals: List[Any]) -> Env:
    new = dict(env)
    for k, v in zip(keys, vals):
        new[k] = v
    return new

# -----------------------------
# Simplifiers
# -----------------------------

def simplify_nary(op_sym: Symbol, args: List[SExpr]) -> SExpr:
    name = op_sym.name
    if name == "+":
        flat: List[SExpr] = []
        for a in args:
            if isinstance(a, list) and a and a[0] == S("+"):
                flat.extend(a[1:])
            else:
                flat.append(a)
        flat = [x for x in flat if not (is_number(x) and x == 0)]
        if len(flat) == 0:
            return 0
        if len(flat) == 1:
            return flat[0]
        return [S("+")] + flat
    elif name == "*":
        flat = []
        for a in args:
            if isinstance(a, list) and a and a[0] == S("*"):
                flat.extend(a[1:])
            else:
                flat.append(a)
        if any(is_number(x) and x == 0 for x in flat):
            return 0
        flat = [None if (is_number(x) and x == 1) else x for x in flat]
        flat = [x for x in flat if x is not None]
        if len(flat) == 0:
            return 1
        if len(flat) == 1:
            return flat[0]
        return [S("*")] + flat
    else:
        return [op_sym] + args

# -----------------------------
# Evaluator with partial simplification
# -----------------------------

@dataclass
class EvalResult:
    value: Any
    form: SExpr
    is_value: bool


def eval_simplify(form: SExpr, env: Env) -> EvalResult:
    # literals
    if literal_value(form):
        return EvalResult(form, form, True)

    # symbols
    if is_symbol(form):
        if form in env:
            v = env[form]
            return EvalResult(v, v, True)
        return EvalResult(None, form, False)

    # lists
    if isinstance(form, list):
        if not form:
            return EvalResult(None, form, False)
        head, *tail = form

        # (fn [params] body)
        if is_symbol(head) and head.name == "fn":
            params = tail[0]
            body = tail[1]
            return EvalResult(Closure(params, body, env), form, True)

        # (if c a b)
        if is_symbol(head) and head.name == "if":
            c = eval_simplify(tail[0], env)
            a = eval_simplify(tail[1], env)
            b = eval_simplify(tail[2], env)
            if c.is_value:
                return a if c.value else b
            return EvalResult(None, [S("if"), c.form, a.form, b.form], False)

        # arithmetic
        if is_symbol(head) and head.name in {"+", "-", "*", "/"}:
            evs = [eval_simplify(a, env) for a in tail]
            if all(e.is_value for e in evs):
                vals = [e.value for e in evs]
                try:
                    if head.name == "+":
                        v = reduce(op.add, vals)
                    elif head.name == "-":
                        v = -vals[0] if len(vals) == 1 else reduce(op.sub, vals)
                    elif head.name == "*":
                        v = reduce(op.mul, vals)
                    else:
                        v = 1 / vals[0] if len(vals) == 1 else reduce(op.truediv, vals)
                    return EvalResult(v, v, True)
                except Exception:
                    return EvalResult(None, [head] + vals, False)
            forms = [e.form for e in evs]
            if head.name in {"+", "*"}:
                return EvalResult(None, simplify_nary(head, forms), False)
            return EvalResult(None, [head] + forms, False)

        # comparisons
        if is_symbol(head) and head.name in {"<", "<=", ">", ">=", "="}:
            evs = [eval_simplify(a, env) for a in tail]
            if all(e.is_value for e in evs):
                vals = [e.value for e in evs]
                ops = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge, "=": op.eq}
                # Hy comparisons chain like Python; we support 2-arg here for simplicity
                if len(vals) == 2:
                    v = ops[head.name](vals[0], vals[1])
                else:
                    v = all(ops[head.name](vals[i], vals[i + 1]) for i in range(len(vals) - 1))
                return EvalResult(v, v, True)
            return EvalResult(None, [head] + [e.form for e in evs], False)

        # HOFs: compose, partial, map, filter, reduce
        if is_symbol(head) and head.name == "compose":
            f = eval_simplify(tail[0], env)
            g = eval_simplify(tail[1], env)
            if f.is_value and g.is_value and is_closure(f.value) and is_closure(g.value):
                # Represent composition as a closure that, when applied, applies f(g(x)).
                x = S("x")
                composed_body = [f.value, [g.value, x]]
                return EvalResult(Closure([x], composed_body, {}), [S("compose"), f.value, g.value], True)
            return EvalResult(None, [S("compose"), f.form, g.form], False)

        if is_symbol(head) and head.name == "partial":
            f = eval_simplify(tail[0], env)
            args = [eval_simplify(a, env) for a in tail[1:]]
            if f.is_value and is_closure(f.value):
                known = [a.value if a.is_value else a.form for a in args]
                # Use a small object for partials
                part = {"type": "partial", "fn": f.value, "known": known}
                return EvalResult(part, [S("partial"), f.form] + [a.form for a in args], True)
            return EvalResult(None, [S("partial"), f.form] + [a.form for a in args], False)

        if is_symbol(head) and head.name == "map":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list):
                out: List[Any] = []
                for v in lst.value:
                    ap = eval_simplify([f.value, v], env)
                    out.append(ap.value if ap.is_value else ap.form)
                return EvalResult(out, out, all(literal_value(x) for x in out))
            return EvalResult(None, [S("map"), f.form, lst.form], False)

        if is_symbol(head) and head.name == "filter":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list):
                out = []
                for v in lst.value:
                    ap = eval_simplify([f.value, v], env)
                    # if predicate unknown, conservatively keep item but as value
                    if ap.is_value:
                        if ap.value:
                            out.append(v)
                    else:
                        out.append(v)
                return EvalResult(out, out, True)
            return EvalResult(None, [S("filter"), f.form, lst.form], False)

        if is_symbol(head) and head.name == "reduce":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            has_init = len(tail) > 2
            init = eval_simplify(tail[2], env) if has_init else None
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list) and (not has_init or init.is_value):
                if has_init:
                    cur = init.value
                    rest = lst.value
                else:
                    if not lst.value:
                        return EvalResult(None, [S("reduce"), f.form, lst.form], False)
                    cur = lst.value[0]
                    rest = lst.value[1:]
                for v in rest:
                    ap = eval_simplify([f.value, cur, v], env)
                    cur = ap.value if ap.is_value else ap.form
                return EvalResult(cur, cur, literal_value(cur))
            forms = [f.form, lst.form] + ([init.form] if has_init and init else [])
            return EvalResult(None, [S("reduce")] + forms, False)

        # Function application: (f a b ...)
        fun = eval_simplify(head, env)
        argres = [eval_simplify(a, env) for a in tail]
        arg_vals = [e.value for e in argres]
        arg_forms = [e.form for e in argres]

        # calling a closure
        if fun.is_value and is_closure(fun.value):
            cl: Closure = fun.value
            n = len(cl.params)
            k = min(n, len(arg_vals))
            new_env = env_extend(cl.env, cl.params[:k], arg_vals[:k])
            leftover = cl.params[k:]
            if not leftover:
                return eval_simplify(cl.body, new_env)
            # currying: too few args -> closure with fewer params
            return EvalResult(Closure(leftover, cl.body, new_env), [S("partial"), fun.form, S(":TOO-FEW-ARGS")], True)

        # calling a partial
        if fun.is_value and isinstance(fun.value, dict) and fun.value.get("type") == "partial":
            base: Closure = fun.value["fn"]
            known = fun.value["known"]
            merged = known + arg_vals
            return eval_simplify([base] + merged, env)

        # not enough info
        return EvalResult(None, [fun.form] + arg_forms, False)

    # fallback
    return EvalResult(None, form, False)

# -----------------------------
# Random expression generator
# -----------------------------

def rand_num() -> Union[int, float]:
    if random.random() < 0.5:
        return random.randint(-5, 5)
    return random.randint(-9, 9) / 2.0

def rand_bool() -> bool:
    return random.choice([True, False])

def rand_list() -> List[SExpr]:
    return [rand_num() for _ in range(random.randint(1, 5))]

def rand_var(allow_frees: bool) -> Symbol:
    return random.choice(FREE_SYMS) if allow_frees else S("u")

def rand_prim(allow_frees: bool) -> SExpr:
    return random.choice([rand_num(), rand_bool(), rand_list(), rand_var(allow_frees)])

def rand_op() -> Symbol:
    return random.choice([S("+"), S("*"), S("-"), S("/")])

def rand_cmp() -> Symbol:
    return random.choice([S("="), S("<"), S(">"), S("<="), S(">=")])

def gen_expr(depth: int, allow_frees: bool) -> SExpr:
    if depth <= 0:
        return rand_prim(allow_frees)
    kind = random.randint(0, 9)
    if kind <= 2:  # arithmetic
        op_sym = rand_op()
        a = gen_expr(depth - 1, allow_frees)
        b = gen_expr(depth - 1, allow_frees)
        return [op_sym, a, b]
    if kind == 3:  # comparison
        op_sym = rand_cmp()
        a = gen_expr(depth - 1, allow_frees)
        b = gen_expr(depth - 1, allow_frees)
        return [op_sym, a, b]
    if kind == 4:  # if
        c = [rand_cmp(), gen_expr(depth - 1, allow_frees), gen_expr(depth - 1, allow_frees)]
        a = gen_expr(depth - 1, allow_frees)
        b = gen_expr(depth - 1, allow_frees)
        return [S("if"), c, a, b]
    if kind == 5:  # lambda
        p = random.choice([S("v"), S("w"), S("u")])
        body = gen_expr(depth - 1, allow_frees)
        return [S("fn"), [p], body]
    if kind == 6:  # application
        f = gen_expr(depth - 1, allow_frees)
        a = gen_expr(depth - 1, allow_frees)
        return [f, a]
    if kind == 7:  # compose
        f = gen_expr(depth - 1, allow_frees)
        g = gen_expr(depth - 1, allow_frees)
        return [S("compose"), f, g]
    if kind == 8:  # partial
        f = gen_expr(depth - 1, allow_frees)
        a = gen_expr(depth - 1, allow_frees)
        return [S("partial"), f, a]
    # map/filter/reduce
    hof = random.choice([S("map"), S("filter"), S("reduce")])
    f = gen_expr(depth - 1, allow_frees)
    lst = rand_list()
    if hof.name == "reduce":
        return [hof, f, lst, rand_prim(allow_frees)]
    return [hof, f, lst]

base_env: Env = {
    S("inc"): Closure([S("x")], [S("+"), S("x"), 1], {}),
    S("square"): Closure([S("x")], [S("*"), S("x"), S("x")], {}),
    S("add"): Closure([S("a"), S("b")], [S("+"), S("a"), S("b")], {}),
    S("mul"): Closure([S("a"), S("b")], [S("*"), S("a"), S("b")], {}),
}
# -----------------------------
# Driver
# -----------------------------
def calcfreeval(s:SExpr)->int:
    pass
    return 10

def geneval(N: int, max_depth: int, seed: int | None, allow_frees: bool,max_bind:int=3,show: bool=False) -> tuple:
    if seed is not None:
        random.seed(seed)
    sexp_eval=[]
    for _ in range(N):
        expr = gen_expr(max_depth, allow_frees)
        res = eval_simplify(expr, base_env)
        if(show):
            if res.is_value:
                print(f"Expr: {pprint_sexpr(expr)}=> {res.value}")
            else:
                print(f"Expr: {pprint_sexpr(expr)}=> {res.form}")
        if res.is_value:
            sexp_eval.append((pprint_sexpr(expr),res.value))
        else:
            sexp_eval.append((pprint_sexpr(expr),res.form))
    if(allow_frees):
        sexp_eval= [(s,v) for s,v in sexp_eval if calcfreeval(s)<=max_bind]
    return zip(sexp_eval)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random higher‑order S‑expr generator + partial evaluator (Python + hy models)")
    p.add_argument("--n", type=int, default=10, help="number of expressions")
    p.add_argument("--max-depth", type=int, default=4, help="maximum nesting depth")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--closed", action="store_true", help="disallow free variables in generation")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    geneval(ns.n, ns.max_depth, ns.seed, allow_frees=not ns.closed)
