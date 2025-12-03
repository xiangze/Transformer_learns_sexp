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

Now supports **list construction/evaluation** forms:
- `(list a b c ...)`  — build list from items (evaluates items; partial if unknown)
- `(cons a xs)`      — prepend element `a` to list `xs`
- `(first xs)`       — first element (partial if empty/unknown)
- `(rest xs)`        — list without first (empty → `[]`)
- `(append xs ys)`   — concatenate lists
- `(len xs)`         — length if list is known

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
from random import choice,randint
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Set, Tuple, Union
import sys
from hy.models import Symbol  # data structure usage only

SExpr = Union[int, float, bool, Symbol, List["SExpr"]]
Env = Dict[Symbol, Any]

# -----------------------------
# Utility predicates
# -----------------------------
is_number=lambda x:isinstance(x, (int, float)) and not isinstance(x, bool)
is_bool=lambda x:isinstance(x, bool)
is_symbol=lambda x:isinstance(x, Symbol)
is_closure =lambda v:isinstance(v, Closure)

def is_list_literal(x: Any) -> bool:
    return isinstance(x, list) and (not x or not (is_symbol(x[0]) and f"{x[0]}" in {
        'fn','if','+','-','*','/','<','<=','>','>=','=','compose','partial','map','filter','reduce',
        'list','cons','first','rest','append','len' }))


def literal_value(x: Any) -> bool:
    if is_number(x) or is_bool(x):
        return True
    if is_list_literal(x):
        return all(literal_value(e) for e in x)
    return False

#SExpr
S=lambda name:Symbol(name)

_DEFAULT_POOL = [S("x"), S("y"), S("z"), S("t")]

# -----------------------------
# Closures(環境つき式)
# -----------------------------
@dataclass
class Closure:
    params: List[Symbol]
    body: SExpr
    env: Env

# -----------------------------
# Pretty printer
# -----------------------------
FUNCS={'fn','if','+','-','*','/','<','<=','>','>=','=','compose','partial','map','filter','reduce',
       'list','cons','first','rest','append','len' }
def pprint_sexpr(form: SExpr) -> str:
    if is_symbol(form):
        return f"{form}"
    elif is_number(form) or is_bool(form):
        return str(form)
    elif isinstance(form, list):
        xs=" ".join(pprint_sexpr(x) for x in form)
        xl=",".join(pprint_sexpr(x) for x in form)
        return f"({xs})" if (form and isinstance(form[0], Symbol) and f"{form[0]}" in FUNCS) else f"[{xl}]" 
    elif isinstance(form, Closure):
        #return f"(Closure params={form.params} body={pprint_sexpr(form.body)})"
        return f"(fn [{form.params}] {pprint_sexpr(form.body)})"
        # (fn [params] body) — constructing a closure is not a simplification step
    else:
        return str(form)

# -----------------------------
# Environment helpers
# -----------------------------
def env_extend(env: Env, keys: List[Symbol], vals: List[Any]) -> Env:
    return  dict(env).update({k:v for k,v in zip(keys, vals)})

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
    elif isinstance(expr, list) and expr:
        head, *tail = expr
        if is_symbol(head) and f"{head}" == "fn":
            params ,body= tail[:2]
            new_bound = set(bound) | set(params)
            return free_vars(body, env, new_bound)
        # generic: union across subforms
        for e in expr:
            out |= free_vars(e, env, bound)
    return out

# -----------------------------
# Simplifiers
# -----------------------------
#2項演算
def simplify_nary(op_sym: Symbol, args: List[SExpr]) -> Tuple[SExpr, int]:
    """Return (simplified_form, steps_added)."""
    name = f"{op_sym}"
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
        elif len(flat2) == 1:
            return flat2[0], int(pprint_sexpr(before) != pprint_sexpr(flat2[0]))
        else:
            after = [S("+")] + flat2
            return after, int(pprint_sexpr(before) != pprint_sexpr(after))
    elif name == "*":
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
        elif len(flat2) == 1:
            return flat2[0], int(pprint_sexpr(before) != pprint_sexpr(flat2[0]))
        else:
            after = [S("*")] + flat2
            return after, int(pprint_sexpr(before) != pprint_sexpr(after))
    else:
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

def is_valuelist(xs):
     try:
         return isinstance(xs.value, list) and xs.is_value
     except AttributeError:
         return False

def is_valueclosure(xs):
    try:
        return isinstance(xs.value, Closure) and xs.is_value
    except AttributeError:
        return False

RemainResult=lambda sexp,steps :EvalResult(None,sexp,False,steps)
SuccResult=lambda val,sexp,steps,success=True,nstep=1 :EvalResult(val,sexp,success,steps+nstep)

### (partial) simplify S-expressions
def eval_simplify(form: SExpr, env: Env) -> EvalResult:
    # literals (numbers, bools, list literals)
    if literal_value(form):
        return EvalResult(form, form, True, 0)
    # symbols
    if is_symbol(form):
        if form in env:
            v = env[form]
            return EvalResult(v, v, True, 0)
        return RemainResult(form,0)
    # lists(S-expressions)
    elif isinstance(form, list):
        if not form:
            return RemainResult(form,0)    
        head, *tail = form
        head_name=f"{head}"
        # (fn [params] body) — constructing a closure is not a simplification step
        if is_symbol(head):
            if(head_name == "fn"):
                params,body = tail[:2]
                return EvalResult(Closure(params, body, env), form, True, 0)
         # (if c a b)
            elif head_name == "if":
                c,a,b=[eval_simplify(t, env) for t in tail[:3]]
                steps = _sum_steps(c, a, b)
                if c.is_value:
                    # one step for folding the conditional
                    if c.value:
                        return EvalResult(a.value, a.form, a.is_value, steps+1)
                    else:
                        return EvalResult(b.value, b.form, b.is_value, steps+1)
                else:
                    return RemainResult([S("if"), c.form, a.form, b.form], steps)
            # arithmetic
            elif head_name in {"+", "-", "*", "/"}:
                evs = [eval_simplify(a, env) for a in tail]
                if(evs!=[]):
                    steps = _sum_steps(*evs)
                    if all(e.is_value for e in evs):
                        vals = [e.value for e in evs]
                        try:
                            if head_name == "+":
                                v = reduce(op.add, vals)
                            elif head_name == "-":
                                v = -vals[0] if len(vals) == 1 else reduce(op.sub, vals)
                            elif head_name == "*":
                                v = reduce(op.mul, vals)
                            else:# head_name == "/":
                                v = 1 / vals[0] if len(vals) == 1 else reduce(op.truediv, vals)
                            return SuccResult(v, v,steps)
                        except Exception:
                            return RemainResult([head] + vals, steps)
                    forms = [e.form for e in evs]
                    if head_name in {"+", "*"}:
                        simplified, add = simplify_nary(head, forms)
                        return RemainResult(simplified, steps + add)
                return RemainResult([head] + forms, steps)
            # comparisons
            elif  head_name in {"<", "<=", ">", ">=", "="}:
                #'list' and 'float
                evs = [eval_simplify(a, env) for a in tail]
                steps = _sum_steps(*evs)
                if all(e.is_value for e in evs):
                    vals = [e.value for e in evs]
                    ops = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge, "=": op.eq}
                    comp=ops[head_name]
                    if len(vals) == 2:
                        v = comp(vals[0], vals[1])
                    else:
                        v = all(comp(vals[i], vals[i+1]) for i in range(len(vals) - 1))
                    return SuccResult(v, v,steps)
                else:
                    return RemainResult([head] + [e.form for e in evs], steps)
            # -----------------
            # List forms
            # -----------------
            elif head_name == "list":
                items = [eval_simplify(a, env) for a in tail]
                steps = _sum_steps(*items)
                if all(it.is_value for it in items):
                    lst = [it.value for it in items]
                    return SuccResult(lst,lst,steps)
                return RemainResult([S("list")] + [it.form for it in items], steps)
            elif head_name == "cons":
                a = eval_simplify(tail[0], env)
                xs = eval_simplify(tail[1], env)
                steps = _sum_steps(a, xs)
                if a.is_value and is_valuelist(xs):
                    lst = [a.value] + xs.value
                    return SuccResult(lst,lst,steps)
                return RemainResult([S("cons"), a.form, xs.form], steps)
            elif head_name == "first":
                xs = eval_simplify(tail[0], env)
                steps = _sum_steps(xs)
                if is_valuelist(xs):
                    if len(xs.value) > 0:
                        return EvalResult(xs.value[0], xs.value[0], literal_value(xs.value[0]), steps + 1)
                    return RemainResult([S("first"), []], steps)
                return RemainResult([S("first"), xs.form], steps)

            elif head_name == "rest":
                xs = eval_simplify(tail[0], env)
                steps = _sum_steps(xs)
                if is_valuelist(xs):
                    return SuccResult(xs.value[1:], xs.value[1:], steps)
                return RemainResult([S("rest"), xs.form], steps)

            elif head_name == "append":
                xs = eval_simplify(tail[0], env)
                ys = eval_simplify(tail[1], env)
                steps = _sum_steps(xs, ys)
                if is_valuelist(xs) and is_valuelist(ys):
                    lst = xs.value + ys.value
                    return EvalResult(lst, lst, True, steps + 1)
                return RemainResult([S("append"), xs.form, ys.form], steps)
            elif head_name == "len":
                xs = eval_simplify(tail[0], env)
                steps = _sum_steps(xs)
                if is_valuelist(xs):
                    return SuccResult(len(xs.value), len(xs.value), steps)
                return RemainResult([S("len"), xs.form], steps)
            # HOFs: compose, partial, map, filter, reduce
            elif head_name == "compose":
                f,g= [eval_simplify(t, env) for t in tail[:2]]
                steps = _sum_steps(f, g)
                if f.is_value and g.is_value and is_closure(f.value) and is_closure(g.value):
                    x = S("x")
                    composed_body = [f.value, [g.value, x]]
                    return SuccResult(Closure([x], composed_body, {}), [S("compose"), f.value, g.value], steps)
                else:
                    return RemainResult([S("compose"), f.form, g.form], steps)
            elif  head_name == "partial":
                f = eval_simplify(tail[0], env)
                args = [eval_simplify(a, env) for a in tail[1:]]
                steps = _sum_steps(f, *args)
                if is_valueclosure(f):
                    known = [a.value if a.is_value else a.form for a in args]
                    part = {"type": "partial", "fn": f.value, "known": known}
                    return SuccResult(part, [S("partial"), f.form] + [a.form for a in args], steps + 1)
                return RemainResult([S("partial"), f.form] + [a.form for a in args], steps)
            elif head_name == "map":
                f,lst = [eval_simplify(t, env) for t in tail[:2]]
                steps = _sum_steps(f, lst)
                if (lst != []) and is_valuelist(lst) and is_valueclosure(f):
                    out: List[Any] = []
                    add_steps = 1  # recognizing & executing map
                    try:
                        for v in lst.value:
                            ap = eval_simplify([f.value, v], env)
                            out.append(ap.value if ap.is_value else ap.form)
                            add_steps += ap.steps
                        return EvalResult(out, out, all(literal_value(x) for x in out), steps + add_steps)
                    except Exception:
                        return RemainResult([S("map"), f.form, lst.form],steps)
                return RemainResult([S("map"), f.form, lst.form],steps)
            elif  head_name == "filter":
                f,lst = [eval_simplify(t, env) for t in tail[:2]]
                steps = _sum_steps(f, lst)
                if is_valuelist(lst) and is_valueclosure(f):
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
                else:
                    return RemainResult([S("filter"), f.form, lst.form],steps)
            elif  head_name == "reduce":
                f,lst = [eval_simplify(t, env) for t in tail[:2]]
                has_init = len(tail) > 2
                init = eval_simplify(tail[2], env) if has_init else None
                steps = _sum_steps(f, lst, *( [init] if init else [] ))
                if is_valuelist(lst) and is_valueclosure(f) and (not has_init or init.is_value):
                    add_steps = 1  # recognizing & starting reduce
                    if has_init:
                        cur = init.value
                        rest = lst.value
                    else:
                        if not lst.value:
                            return RemainResult([S("reduce"), f.form, lst.form],steps)
                        cur = lst.value[0]
                        rest = lst.value[1:]
                    for v in rest:
                        ap = eval_simplify([f.value, cur, v], env)
                        cur = ap.value if ap.is_value else ap.form
                        add_steps += ap.steps + 1  # each fold counts at least one step
                    return EvalResult(cur, cur, literal_value(cur), steps + add_steps)
                forms = [f.form, lst.form] + ([init.form] if has_init and init else [])
                return RemainResult( [S("reduce")] + forms, steps)
            else:
            # Function application: (f a b ...)
                fun = eval_simplify(head, env)
                argres = [eval_simplify(a, env) for a in tail]
                steps = _sum_steps(fun, *argres)
                arg_vals = [e.value for e in argres]
                arg_forms = [e.form for e in argres]
                # calling a closure (beta‑reduction)
                if fun.is_value:
                    if is_closure(fun.value):
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
                elif isinstance(fun.value, dict) and fun.value.get("type") == "partial":
                    base: Closure = fun.value["fn"]
                    known = fun.value["known"]
                    merged = known + arg_vals
                    res = eval_simplify([base] + merged, env)
                    return EvalResult(res.value, res.form, res.is_value, steps + res.steps + 1)
                return RemainResult([fun.form] + arg_forms, steps)
    else:
        # fallback
        return RemainResult(form,0)

# -----------------------------
# Random expression generator
# -----------------------------
rand_num =lambda :randint(-5, 5) if random.random() < 0.5 else random.randint(-9, 9) / 2.0
rand_bool=lambda:choice([True, False])
rand_list=lambda:[rand_num() for _ in range(randint(1, 5))]

def rand_list_literal(allow_frees: bool, var_pool: List[Symbol]) -> List[SExpr]:
    # mix numbers, bools, and maybe symbols
    L = random.randint(0, 5)
    candidates: List[SExpr] = [rand_num(), rand_bool()]
    if allow_frees and var_pool:
        candidates.append(random.choice(var_pool))
    return [random.choice(candidates) for _ in range(L)]

def rand_var(allow_frees: bool, var_pool: List[Symbol]) -> Symbol:
    return random.choice(var_pool) if (allow_frees and var_pool) else S("u")

def rand_prim(allow_frees: bool, var_pool: List[Symbol]) -> SExpr:
    return random.choice([rand_num(), rand_bool(), rand_list_literal(allow_frees, var_pool), rand_var(allow_frees, var_pool)])
#S:Symbol
rand_op=lambda :choice([S("+"), S("*"), S("-"), S("/")])
rand_cmp=lambda :choice([S("="), S("<"), S(">"), S("<="), S(">=")])

expr_kind={1:"arithmetic",2:"comparison",3:"if",4:"lambda",5:"application",6:"compose",7:"partial",8:"hof",9:"list_literal",10:"cons",11:"first_rest",12:"append_len"}

def gen_expr(depth: int, allow_frees: bool, var_pool: List[Symbol]) -> SExpr:
    if depth <= 0:
        return rand_prim(allow_frees, var_pool)
    kind = randint(0, len(expr_kind))
    a,b = [ gen_expr(depth - 1, allow_frees, var_pool) for _ in range(2)]
    if kind <= 2:  # arithmetic
        op_sym = rand_op()
        return [op_sym, a, b]
    elif kind == 3:  # comparison
        op_sym = rand_cmp()
        return [op_sym, a, b]
    elif kind == 4:  # if
        c = [rand_cmp(), gen_expr(depth - 1, allow_frees, var_pool), gen_expr(depth - 1, allow_frees, var_pool)]
        return [S("if"), c, a, b]
    if kind == 5:  # lambda
        p = choice([S("v"), S("w"), S("u")])
        return [S("fn"), [p], a]
    elif kind == 6:  # application
        return [a, b]
    elif kind == 7:  # compose
        return [S("compose"), a, b]
    elif kind == 8:  # partial
        return [S("partial"), a, b]
    elif kind == 9:  # map/filter/reduce
        hof = choice([S("map"), S("filter"), S("reduce")])    
        f = gen_expr(depth - 1, allow_frees, var_pool)
        lst = rand_list_literal(allow_frees, var_pool)
        if "reduce" in f"{hof}" : #name
            return [hof, f, lst, rand_prim(allow_frees, var_pool)]
        return [hof, f, lst]
    elif kind == 10:  # list literal from forms: (list ...)
        m = random.randint(0, 4)
        items = [gen_expr(depth - 1, allow_frees, var_pool) for _ in range(m)]
        return [S("list")] + items
    elif kind == 11:  # cons
        a = gen_expr(depth - 1, allow_frees, var_pool)
        xs = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("cons"), a, xs]
    elif kind == 12:  # first / rest
        opn = random.choice([S("first"), S("rest")])
        xs = gen_expr(depth - 1, allow_frees, var_pool)
        return [opn, xs]
    # append / len
    elif random() < 0.5:
        xs = gen_expr(depth - 1, allow_frees, var_pool)
        ys = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("append"), xs, ys]
    else:
        xs = gen_expr(depth - 1, allow_frees, var_pool)
        return [S("len"), xs]


# -----------------------------
# Driver
# -----------------------------
def _make_var_pool(size: int) -> List[Symbol]:
    if size <= len(_DEFAULT_POOL):
        return _DEFAULT_POOL[:size]
    extra = [S(f"v{i}") for i in range(size - len(_DEFAULT_POOL))]
    return _DEFAULT_POOL + extra

def geneval(N: int, max_depth: int, allow_frees: bool, var_pool_size: int, target_free: int | None, show:bool=False,show_short=True,seed: int=42) -> list[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    if not allow_frees:
        target_free = 0

    var_pool = _make_var_pool(max(0, var_pool_size))

    base_env: Env = {
        S("inc"): Closure([S("x")], [S("+"), S("x"), 1], {}),
        S("square"): Closure([S("x")], [S("*"), S("x"), S("x")], {}),
        S("add"): Closure([S("a"), S("b")], [S("+"), S("a"), S("b")], {}),
        S("mul"): Closure([S("a"), S("b")], [S("*"), S("a"), S("b")], {}),
    }
    sexp_evals: List[Dict[str, Any]] = []
    max_attempts = 300
    for i in range(N):
        attempts = 0
        expr = None
        fv_set: Set[Symbol] = set()
        while True:
            expr = gen_expr(max_depth, allow_frees, var_pool)
            fv_set = free_vars(expr, base_env)
            if target_free is None or len(fv_set) == target_free:
                break
            attempts += 1
            if attempts >= max_attempts:
                break # accept closest we got
        # If we failed to hit target, pick the closest generated in the loop (simple: keep last)
        try:
            res = eval_simplify(expr, base_env)
            assert(res.steps >= 0)
        except Exception:
            print("# Exception eval:", pprint_sexpr(expr))
            res=RemainResult(expr,0)
            #break

        fv_list = sorted((f"{s}" for s in fv_set))
        result={
            "Expr": pprint_sexpr(expr),
            "free_vars":fv_set,
            "fv_list":fv_list,
            "pool_size":len(var_pool),
            "steps":res.steps,
        }
        if target_free is not None:
            result["attempts"]=attempts
        else:
            result["attempts"]= 0

        if res.is_value:
            result["value"]=res.value
            result["isvalue"]=True
        else:
            result["value"]=pprint_sexpr(res.form)
            result["isvalue"]=False

        sexp_evals.append(result)

        if(show or show_short):
            print_simple(i,result)
            if(not show_short):
                print_result(result)
    return sexp_evals

def print_simple(i,result,file=sys.stdout):
        fv_set=result["free_vars"]
        pool_size=result["pool_size"]
        attempts=result["attempts"]
        steps=result["steps"]
        print(f"{i}: free_vars={len(fv_set)}, pool_size={pool_size} ; attempts={attempts},steps={steps}", file=file)

def print_result(result,file=sys.stdout):
    print("Expr:", pprint_sexpr(result["Expr"]),file=file)
    if(result["isvalue"]):
        print(f"=> ",pprint_sexpr(result["value"]),file=file)
    else:
        print(f"⇝ ",pprint_sexpr(result["value"]),file=file)
    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random higher‑order S‑expr generator + partial evaluator (Python + hy models, with step + free‑var counting + list forms)")
    p.add_argument("--n", type=int, default=10, help="number of expressions")
    p.add_argument("--max-depth", type=int, default=4, help="maximum nesting depth")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--closed", action="store_true", help="disallow free variables in generation (forces 0 free vars)")
    p.add_argument("--var-pool", type=int, default=4, help="size of the free-variable symbol pool (x,y,z,t,v0,...) ")
    p.add_argument("--target-free", type=int, default=None, help="require exactly this many distinct free variables (best-effort)")
    p.add_argument("--show",  action="store_true", help="show result")
    p.add_argument("--show_short",  action="store_true", help="show short result")
    return p.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    geneval(ns.n, ns.max_depth, allow_frees=not ns.closed, var_pool_size=ns.var_pool, target_free=ns.target_free,
            show=ns.show, show_short=ns.show_short, seed=ns.seed)