# -*- coding: utf-8 -*-
"""
Random higher‑order S‑expression generator + evaluator (partial evaluation)
Python implementation using the **hy** library's Symbol type — with
**simplification step counting** and **free‑variable counting/constraint**.

Features
--------
- Randomly generate S‑expressions with higher‑order functions: (fn ...),
  application, compose, partial, map/filter/reduce, if, + - * /, comparisons.
- Evaluate to a concrete value when possible (closed terms / enough info).
- If free variables remain, return a **simplified** S‑expression (partial eval).
- Count and report the number of **simplification/evaluation steps** taken.
- NEW: Control the **variable pool size** used for free variables, and/or target
  the **exact number of distinct free variables** appearing in each generated
  expression; also print the count and which variables were used.

Usage
-----
    pip install hy
    python random_hof_sexpr.py --n 10 --max-depth 4 --seed 0
    python random_hof_sexpr.py --n 5 --closed
    python random_hof_sexpr.py --n 10 --var-pool 6 --target-free 2 --seed 1

Notes
-----
- `--closed` forces 0 free variables regardless of `--target-free`.
- `--var-pool` grows the available symbol set beyond [x y z t] by adding v0,v1,...
- If `--target-free K` is given (and not closed), the generator retries up to a
  small limit to hit exactly K distinct free variables.
"""

from __future__ import annotations
import argparse
import operator as op
import random
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Set, Tuple, Union

from hy.models import Symbol  # data structure usage only

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

_DEFAULT_POOL = [S("x"), S("y"), S("z"), S("t")]

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
# Free-variable analysis
# -----------------------------

def free_vars(expr: SExpr, env: Env, bound: Set[Symbol] | None = None) -> Set[Symbol]:
    """Collect symbols that are *not* bound by lambdas and not present in env."""
    if bound is None:
        bound = set()
    out: Set[Symbol] = set()
    if is_symbol(expr):
        if expr not in bound and expr not in env and not isinstance(env.get(expr), Closure):
            out.add(expr)
        return out
    if isinstance(expr, list) and expr:
        head, *tail = expr
        if is_symbol(head) and head.name == "fn":
            params = tail[0]
            body = tail[1]
            new_bound = set(bound) | set(params)
            return free_vars(body, env, new_bound)
        # generic: union across subforms
        for e in expr:
            out |= free_vars(e, env, bound)
    return out

# -----------------------------
# Simplifiers
# -----------------------------

def simplify_nary(op_sym: Symbol, args: List[SExpr]) -> Tuple[SExpr, int]:
    """Return (simplified_form, steps_added)."""
    name = op_sym.name
    if name == "+":
        flat: List[SExpr] = []
        for a in args:
            if isinstance(a, list) and a and a[0] == S("+"):
                flat.extend(a[1:])
            else:
                flat.append(a)
        before = [S("+")] + list(args)
        flat2: List[SExpr] = [x for x in flat if not (is_number(x) and x == 0)]
        if len(flat2) == 0:
            return 0, int(pprint_sexpr(before) != "0")
        if len(flat2) == 1:
            return flat2[0], int(pprint_sexpr(before) != pprint_sexpr(flat2[0]))
        after = [S("+")] + flat2
        return after, int(pprint_sexpr(before) != pprint_sexpr(after))

    if name == "*":
        flat: List[SExpr] = []
        for a in args:
            if isinstance(a, list) and a and a[0] == S("*"):
                flat.extend(a[1:])
            else:
                flat.append(a)
        before = [S("*")] + list(args)
        if any(is_number(x) and x == 0 for x in flat):
            return 0, int(pprint_sexpr(before) != "0")
        flat = [None if (is_number(x) and x == 1) else x for x in flat]
        flat2 = [x for x in flat if x is not None]
        if len(flat2) == 0:
            return 1, int(pprint_sexpr(before) != "1")
        if len(flat2) == 1:
            return flat2[0], int(pprint_sexpr(before) != pprint_sexpr(flat2[0]))
        after = [S("*")] + flat2
        return after, int(pprint_sexpr(before) != pprint_sexpr(after))

    return [op_sym] + args, 0

# -----------------------------
# Evaluator with partial simplification + step counting
# -----------------------------

@dataclass
class EvalResult:
    value: Any
    form: SExpr
    is_value: bool
    steps: int = 0


def _sum_steps(*results: EvalResult) -> int:
    return sum(r.steps for r in results if isinstance(r, EvalResult))


def eval_simplify(form: SExpr, env: Env) -> EvalResult:
    # literals
    if literal_value(form):
        return EvalResult(form, form, True, 0)

    # symbols
    if is_symbol(form):
        if form in env:
            v = env[form]
            return EvalResult(v, v, True, 0)
        return EvalResult(None, form, False, 0)

    # lists
    if isinstance(form, list):
        if not form:
            return EvalResult(None, form, False, 0)
        head, *tail = form

        # (fn [params] body) — constructing a closure is not a simplification step
        if is_symbol(head) and head.name == "fn":
            params = tail[0]
            body = tail[1]
            return EvalResult(Closure(params, body, env), form, True, 0)

        # (if c a b)
        if is_symbol(head) and head.name == "if":
            c = eval_simplify(tail[0], env)
            a = eval_simplify(tail[1], env)
            b = eval_simplify(tail[2], env)
            steps = _sum_steps(c, a, b)
            if c.is_value:
                # one step for folding the conditional
                return (EvalResult(a.value, a.form, a.is_value, steps + 1)
                        if c.value else
                        EvalResult(b.value, b.form, b.is_value, steps + 1))
            return EvalResult(None, [S("if"), c.form, a.form, b.form], False, steps)

        # arithmetic
        if is_symbol(head) and head.name in {"+", "-", "*", "/"}:
            evs = [eval_simplify(a, env) for a in tail]
            steps = _sum_steps(*evs)
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
                    return EvalResult(v, v, True, steps + 1)  # collapse counts as 1 step
                except Exception:
                    return EvalResult(None, [head] + vals, False, steps)
            forms = [e.form for e in evs]
            if head.name in {"+", "*"}:
                simplified, add = simplify_nary(head, forms)
                return EvalResult(None, simplified, False, steps + add)
            return EvalResult(None, [head] + forms, False, steps)

        # comparisons
        if is_symbol(head) and head.name in {"<", "<=", ">", ">=", "="}:
            evs = [eval_simplify(a, env) for a in tail]
            steps = _sum_steps(*evs)
            if all(e.is_value for e in evs):
                vals = [e.value for e in evs]
                ops = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge, "=": op.eq}
                if len(vals) == 2:
                    v = ops[head.name](vals[0], vals[1])
                else:
                    v = all(ops[head.name](vals[i], vals[i + 1]) for i in range(len(vals) - 1))
                return EvalResult(v, v, True, steps + 1)
            return EvalResult(None, [head] + [e.form for e in evs], False, steps)

        # HOFs: compose, partial, map, filter, reduce
        if is_symbol(head) and head.name == "compose":
            f = eval_simplify(tail[0], env)
            g = eval_simplify(tail[1], env)
            steps = _sum_steps(f, g)
            if f.is_value and g.is_value and is_closure(f.value) and is_closure(g.value):
                x = S("x")
                composed_body = [f.value, [g.value, x]]
                return EvalResult(Closure([x], composed_body, {}), [S("compose"), f.value, g.value], True, steps + 1)
            return EvalResult(None, [S("compose"), f.form, g.form], False, steps)

        if is_symbol(head) and head.name == "partial":
            f = eval_simplify(tail[0], env)
            args = [eval_simplify(a, env) for a in tail[1:]]
            steps = _sum_steps(f, *args)
            if f.is_value and is_closure(f.value):
                known = [a.value if a.is_value else a.form for a in args]
                part = {"type": "partial", "fn": f.value, "known": known}
                return EvalResult(part, [S("partial"), f.form] + [a.form for a in args], True, steps + 1)
            return EvalResult(None, [S("partial"), f.form] + [a.form for a in args], False, steps)

        if is_symbol(head) and head.name == "map":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            steps = _sum_steps(f, lst)
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list):
                out: List[Any] = []
                add_steps = 1  # recognizing & executing map
                for v in lst.value:
                    ap = eval_simplify([f.value, v], env)
                    out.append(ap.value if ap.is_value else ap.form)
                    add_steps += ap.steps
                return EvalResult(out, out, all(literal_value(x) for x in out), steps + add_steps)
            return EvalResult(None, [S("map"), f.form, lst.form], False, steps)

        if is_symbol(head) and head.name == "filter":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            steps = _sum_steps(f, lst)
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list):
                out: List[Any] = []
                add_steps = 1
                for v in lst.value:
                    ap = eval_simplify([f.value, v], env)
                    add_steps += ap.steps
                    if ap.is_value:
                        if ap.value:
                            out.append(v)
                    else:
                        out.append(v)
                return EvalResult(out, out, True, steps + add_steps)
            return EvalResult(None, [S("filter"), f.form, lst.form], False, steps)

        if is_symbol(head) and head.name == "reduce":
            f = eval_simplify(tail[0], env)
            lst = eval_simplify(tail[1], env)
            has_init = len(tail) > 2
            init = eval_simplify(tail[2], env) if has_init else None
            steps = _sum_steps(f, lst, *( [init] if init else [] ))
            if f.is_value and lst.is_value and is_closure(f.value) and isinstance(lst.value, list) and (not has_init or init.is_value):
                add_steps = 1  # recognizing & starting reduce
                if has_init:
                    cur = init.value
                    rest = lst.value
                else:
                    if not lst.value:
                        return EvalResult(None, [S("reduce"), f.form, lst.form], False, steps)
                    cur = lst.value[0]
                    rest = lst.value[1:]
                for v in rest:
                    ap = eval_simplify([f.value, cur, v], env)
                    cur = ap.value if ap.is_value else ap.form
                    add_steps += ap.steps + 1  # each fold counts at least one step
                return EvalResult(cur, cur, literal_value(cur), steps + add_steps)
            forms = [f.form, lst.form] + ([init.form] if has_init and init else [])
            return EvalResult(None, [S("reduce")] + forms, False, steps)

        # Function application: (f a b ...)
        fun = eval_simplify(head, env)
        argres = [eval_simplify(a, env) for a in tail]
        steps = _sum_steps(fun, *argres)
        arg_vals = [e.value for e in argres]
        arg_forms = [e.form for e in argres]

        # calling a closure (beta‑reduction)
        if fun.is_value and is_closure(fun.value):
            cl: Closure = fun.value
            n = len(cl.params)
            k = min(n, len(arg_vals))
            new_env = env_extend(cl.env, cl.params[:k], arg_vals[:k])
            leftover = cl.params[k:]
            if not leftover:
                res = eval_simplify(cl.body, new_env)
                return EvalResult(res.value, res.form, res.is_value, steps + res.steps + 1)
            # currying: too few args -> closure with fewer params (not counted as a rewrite)
            return EvalResult(Closure(leftover, cl.body, new_env), [S("partial"), fun.form, S(":TOO-FEW-ARGS")], True, steps)

        # calling a partial
        if fun.is_value and isinstance(fun.value, dict) and fun.value.get("type") == "partial":
            base: Closure = fun.value["fn"]
            known = fun.value["known"]
            merged = known + arg_vals
            res = eval_simplify([base] + merged, env)
            return EvalResult(res.value, res.form, res.is_value, steps + res.steps + 1)

        # not enough info — return simplified application
        return EvalResult(None, [fun.form] + arg_forms, False, steps)

    # fallback
    return EvalResult(None, form, False, 0)

# -----------------------------
# Random expression generator
# -----------------------------

def rand_num() -> Union[int, float]:
    return random.randint(-5, 5) if random.random() < 0.5 else random.randint(-9, 9) / 2.0


def rand_bool() -> bool:
    return random.choice([True, False])


def rand_list() -> List[SExpr]:
    return [rand_num() for _ in range(random.randint(1, 5))]


def rand_var(allow_frees: bool, var_pool: List[Symbol]) -> Symbol:
    return random.choice(var_pool) if allow_frees else S("u")


def rand_prim(allow_frees: bool, var_pool: List[Symbol]) -> SExpr:
    return random.choice([rand_num(), rand_bool(), rand_list(), rand_var(allow_frees, var_pool)])


def rand_op() -> Symbol:
    return random.choice([S("+"), S("*"), S("-"), S("/")])


def rand_cmp() -> Symbol:
    return random.choice([S("="), S("<"), S(">"), S("<="), S(">=")])


def gen_expr(depth: int, allow_frees: bool, var_pool: List[Symbol]) -> SExpr:
    if depth <= 0:
        return rand_prim(allow_frees, var_pool)
    kind = random.randint(0, 9)
    if kind <= 2:  # arithmetic
        op_sym = rand_op()
        a = gen_expr(depth - 1, allow_frees, var_pool)
        b = gen_expr(depth - 1, allow_frees, var_pool)
        return [op_sym, a, b]
    if kind == 3:  # comparison
        op_sym = rand_cmp()
        a = gen_expr(depth - 1, allow_frees, var_pool)
        b = gen_expr(depth - 1, allow_frees, var_pool)
        return [op_sym, a, b]
    if kind == 4:  # if
        c = [rand_cmp(), gen_expr(depth - 1, allow_frees, var_pool), gen_expr(depth - 1, allow_frees, var_pool)]
        a = gen_expr(depth - 1, allow_frees, var_pool)
        b = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("if"), c, a, b]
    if kind == 5:  # lambda
        p = random.choice([S("v"), S("w"), S("u")])
        body = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("fn"), [p], body]
    if kind == 6:  # application
        f = gen_expr(depth - 1, allow_frees, var_pool)
        a = gen_expr(depth - 1, allow_frees, var_pool)
        return [f, a]
    if kind == 7:  # compose
        f = gen_expr(depth - 1, allow_frees, var_pool)
        g = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("compose"), f, g]
    if kind == 8:  # partial
        f = gen_expr(depth - 1, allow_frees, var_pool)
        a = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("partial"), f, a]
    # map/filter/reduce
    hof = random.choice([S("map"), S("filter"), S("reduce")])
    f = gen_expr(depth - 1, allow_frees, var_pool)
    lst = rand_list()
    if hof.name == "reduce":
        return [hof, f, lst, rand_prim(allow_frees, var_pool)]
    return [hof, f, lst]

# -----------------------------
# Driver
# -----------------------------

def _make_var_pool(size: int) -> List[Symbol]:
    if size <= len(_DEFAULT_POOL):
        return _DEFAULT_POOL[:size]
    extra = [S(f"v{i}") for i in range(size - len(_DEFAULT_POOL))]
    return _DEFAULT_POOL + extra


def geneval(N: int, max_depth: int, seed: int | None, allow_frees: bool, var_pool_size: int, target_free: int | None,show:bool=False)->list :
    if seed is not None:
        random.seed(seed)

    # Closed wins over target_free
    if not allow_frees:
        target_free = 0

    var_pool = _make_var_pool(max(0, var_pool_size))

    base_env: Env = {
        S("inc"): Closure([S("x")], [S("+"), S("x"), 1], {}),
        S("square"): Closure([S("x")], [S("*"), S("x"), S("x")], {}),
        S("add"): Closure([S("a"), S("b")], [S("+"), S("a"), S("b")], {}),
        S("mul"): Closure([S("a"), S("b")], [S("*"), S("a"), S("b")], {}),
    }
    sexp_eval=[]
    for _ in range(N):        # Try to satisfy target_free if provided
        attempts = 0
        expr = None
        fv_set: Set[Symbol] = set()
        max_attempts = 300
        while True:
            expr = gen_expr(max_depth, allow_frees, var_pool)
            fv_set = free_vars(expr, base_env)
            if target_free is None or len(fv_set) == target_free:
                break
            attempts += 1
            if attempts >= max_attempts:
                break  # accept closest we got
        # If we failed to hit target, pick the closest generated in the loop (simple: keep last)

        res = eval_simplify(expr, base_env)
        fv_list = sorted((s.name for s in fv_set))

        result={
            "Expr": pprint_sexpr(expr),
            "free_vars":fv_set,
            "fv_list":fv_list,
            "pool_size":len(var_pool),
            "steps":res.steps
        }

        if target_free is not None:
            result["attempts"]=attempts
        else:
            result["attempts"]= -1

        if res.is_value:
            result["value"]=res.value
        else:
            result["value"]=pprint_sexpr(res.form)

        sexp_eval.append(result)
        if(show):
            print("Expr:", pprint_sexpr(expr))
            print(f"free_vars={len(fv_set)} {fv_list}; pool_size={len(var_pool)}" + (f"; attempts={attempts}" if target_free is not None else ""))
            if res.is_value:
                print(f"=> {res.value}  (steps={res.steps})")
            else:
                print(f"⇝ {pprint_sexpr(res.form)}  (steps={res.steps})")
            print()

    return sexp_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random higher‑order S‑expr generator + partial evaluator (Python + hy models, with step + free-var counting)")
    p.add_argument("--n", type=int, default=10, help="number of expressions")
    p.add_argument("--max-depth", type=int, default=4, help="maximum nesting depth")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--closed", action="store_true", help="disallow free variables in generation (forces 0 free vars)")
    p.add_argument("--var-pool", type=int, default=4, help="size of the free-variable symbol pool (x,y,z,t,v0,...) ")
    p.add_argument("--target-free", type=int, default=None, help="require exactly this many distinct free variables (best-effort)")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    geneval(ns.n, ns.max_depth, ns.seed, allow_frees=not ns.closed, var_pool_size=ns.var_pool, target_free=ns.target_free)
