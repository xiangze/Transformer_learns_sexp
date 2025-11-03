from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Iterable, Set,Union
import random, math, ast, re, argparse
import hy
import re
import peval_pure as p
from sexpdata import Symbol,  dumps, loads
from replacelist import replace_list

"""
自由/束縛変数つき S式のランダム生成 + β-簡約/Hy評価 + Dyck往復（Vector対応）

- 生成される「プログラム」は基本形:
    (do (setv v1 e1) (setv v2 e2) ... (setv vk ek) <final-expr>)
  ここで ek や final-expr は λ抽象/適用/数値式/比較/if を含み得る。
  Hy 評価ではそのまま評価、β-簡約では (let [v1 e1 ... vk ek] final) に変換してから λ計算。

- Dyck＋labels は LIST に加えて VEC（[...]）もサポート。
"""
SExp = Union[list, tuple, str, int, float, bool] #Any?
SYM = Symbol
PRIMS = {
    "+", "-", "*", "/", "pow", "abs", "round", "max", "min",
    "<", "<=", ">", ">=", "=", "!=", "if", "list", "do", "setv", "let", "fn",
}
VAR_POOL = ["x","y","z","u","v","a","b","c","t"]

MOD=7
USE_LET=False

def is_number_atom(x: Any) -> bool: return isinstance(x, (int, float))
def is_symbol(x: Any, name: str) -> bool: return isinstance(x, Symbol) and str(x) == name
def sexp_list(*xs: Any) -> list: return list(xs)

# -----------------------------
# 0) 乱数式（自由/束縛あり）
# -----------------------------
_NUM_UNARY_FUNCS = ["abs"] #, "round"]
_NUM_BINARY_OPS  = ["+", "-", "*"] #, "/"]
_NUM_NARY_FUNCS  = ["max", "min", "sum"]  # sum は (sum (list ...)) として生成
_POW_FUNC        = "pow"
_CMP_OPS         = ["<", "<=", ">", ">=", "=", "!="]

if(MOD==0):
    _NUM_BINARY_OPS.append("/")

def _rand_number() -> float | int:
    if random.random() < 0.6:
        return random.randint(-9, 9)
    return round(random.uniform(-9.0, 9.0), 2)

def fresh_var(used: Set[str]) -> str:
    for _ in range(100):
        v = random.choice(VAR_POOL)
        if v not in used:
            return v
    # いっぱいなら添字を振る
    base = random.choice(VAR_POOL)
    i = 1
    while f"{base}{i}" in used: i += 1
    return f"{base}{i}"

def gen_var_atom(bound: Set[str], allow_free: bool = True) -> Symbol:
    pool = set(bound)
    if allow_free and (not pool or random.random() < 0.5):
        pool.add(random.choice(VAR_POOL))
    return SYM(random.choice(list(pool)))

def gen_numeric_expr(depth: int, bound: Set[str],rtable:list=[0.28,0.40,0.50,0.60,0.75,0.88]) -> Any:
    """数値へ評価される式（変数/λ適用含む）"""
    if depth <= 0 or random.random() < 0.25:
        if random.random() < 0.5 and (bound or random.random() < 0.5):
            return gen_var_atom(bound, allow_free=True)
        return _rand_number()

    r = random.random()
    if r < rtable[0]:
        # 二項
        op = random.choice(_NUM_BINARY_OPS)
        return [SYM(op), gen_numeric_expr(depth-1, bound), gen_numeric_expr(depth-1, bound)]
    elif r < rtable[1]:
        # 単項
        f = random.choice(_NUM_UNARY_FUNCS)
        return [SYM(f), gen_numeric_expr(depth-1, bound)]
    elif r < rtable[2]:
        # pow
        return [SYM(_POW_FUNC), gen_numeric_expr(depth-1, bound), random.randint(-4, 4) or 1]
    elif r < rtable[3]:
        # n-ary
        f = random.choice(_NUM_NARY_FUNCS)
        n = random.randint(2, 4)
        items = [gen_numeric_expr(depth-1, bound) for _ in range(n)]
        return [SYM(f), [SYM("list"), *items]]
    elif r < rtable[4]:
        # if 条件式
        a = gen_numeric_expr(depth-1, bound)
        b = gen_numeric_expr(depth-1, bound)
        cmp = [SYM(random.choice(_CMP_OPS)), a, b]
        th = gen_numeric_expr(depth-1, bound)
        el = gen_numeric_expr(depth-1, bound)
        return [SYM("if"), cmp, th, el]
    elif r < rtable[5] or (not USE_LET):
        # λ抽象→即時適用 ((fn [p] body) arg)
        used = set(bound)
        p = fresh_var(used)
        body = gen_numeric_expr(depth-1, bound | {p})
        lam = [SYM("fn"), [SYM(p)], body]
        arg = gen_numeric_expr(depth-1, bound)
        return [lam, arg]
    elif(USE_LET):
        # let 束縛 (let [v e] body)
        v = fresh_var(set(bound))
        e = gen_numeric_expr(depth-1, bound)
        body = gen_numeric_expr(depth-1, bound | {v})
        return [SYM("let"), [SYM(v), e], body]

def gen_program_with_setv(max_bind: int = 2, max_depth: int = 4) -> Any:
    """
    (do (setv v1 e1) ... (setv vk ek) final)
    で、final は数値に評価されやすい式。
    """
    k = random.randint(0, max_bind)
    bound: Set[str] = set()
    forms: List[Any] = []
    for _ in range(k):
        v = fresh_var(bound)
        # 変数へ λ抽象または数値式を割り当てる
        if random.random() < 0.5:
            # λ抽象
            p = fresh_var(bound | {v})
            body = gen_numeric_expr(max_depth-1, bound | {p})
            e = [SYM("fn"), [SYM(p)], body]
        else:
            e = gen_numeric_expr(max_depth-1, bound)
        forms.append([SYM("setv"), SYM(v), e])
        bound.add(v)

    final = gen_numeric_expr(max_depth, bound)
    return [SYM("do"), *forms, final]

def  replace_sexp(sexp):
    s= re.sub(r"\(fn \(([\w+])\)" ,r"(fn [\1]",dumps(sexp))
    s= replace_list(s)
    if(USE_LET):
        s= re.sub(r"\(let \(([\w+])\)" ,r"(let [\1]",s)
    return s

def  replace_sexp_sed(sexp):
    s= replace_sexp(sexp).replace("\"","").replace("hy.models.","")
    s=s.replace("Integer","").replace("Float","")
    return s

def gen_program_with_setv_s(max_bind: int = 2, max_depth: int = 4)->str:
    return  replace_sexp(gen_program_with_setv(max_bind, max_depth))

# -----------------------------
# 1) Hy で逐次評価
# -----------------------------
def hy_eval_program_str(program_str: str) -> Any:
    # Hy は 1式モデルが基本なので (do ...) に包んでおくと評価が容易
    model = hy.read(program_str)    # 1式
    try:
        return hy.eval(model)
    except NameError as e:
        v=f"{e}".split("'")[1]
        return  replace_sexp_sed(p.peval_except(v,model)) #部分評価
    except Exception as e:
        print(e,";",program_str)
        return None
# -----------------------------
# 2) β-簡約（簡易 λ計算）評価
#    - (let [x a y b] body) を ((fn [x y] body) a b) へ
#    - (do (setv x a) ... final) を (let [x a ...] final) へ
#    - 左最外優先 β-簡約 + プリミティブ数値演算
# -----------------------------
def is_list(x): return isinstance(x, list)
def to_str(l:list): return [str(l) for l in list]
# params = [str(s) for s in vec_to_list(expr[1])]

def free_vars(expr: Any, bound: Set[str] = None) -> Set[str]:
    if bound is None: bound = set()
    if isinstance(expr, Symbol):
        s = str(expr)
        return set() if (s in PRIMS) else ({s} - bound)
    elif is_number_atom(expr) or isinstance(expr, bool) or expr is None:
        return set()
    elif is_list(expr):
        if expr and isinstance(expr[0], Symbol) and str(expr[0]) == "defn": #fn
            # (fn [x ...] body)
            params = to_str(expr[1])
            return free_vars(expr[2], bound | set(params))
        elif expr and isinstance(expr[0], Symbol) and str(expr[0]) == "let":
            # (let [x a y b] body)
            #items = vec_to_list(expr[1])
            items = expr[1]
            bs = {}
            for i in range(0, len(items), 2):
                bs[str(items[i])] = items[i+1]
            out = set()
            # 右辺は旧boundで
            for v, e in bs.items():
                out |= free_vars(e, bound)
            # body は新しい bound で
            out |= free_vars(expr[2], bound | set(bs.keys()))
            return out
        else:
            out = set()
            for x in expr:
                out |= free_vars(x, bound)
            return out
    else:
        return set()

def alpha_rename(body: Any, old: str, new: str) -> Any:
    if isinstance(body, Symbol):
        return SYM(new) if str(body) == old else body
    elif is_list(body):
        if body and is_symbol(body[0], "fn"):
            params = [str(s) for s in body[1]]
            if old in params:
                return body  # シャドウされていたら何もしない
            return [body[0], body[1], alpha_rename(body[2], old, new)]
        else:
            return [alpha_rename(x, old, new) for x in body]
    else:
        return body


def substitute(expr: Any, var: str, val: Any) -> Any:
    """捕獲回避付き置換 [var := val]expr"""
    if isinstance(expr, Symbol):
        return val if str(expr) == var else expr
#    elif is_vec(expr):
#        return list_to_vec(substitute(x, var, val) for x in expr)
    elif is_list(expr):
        if expr and is_symbol(expr[0], "fn"):
            #params = [str(s) for s in vec_to_list(expr[1])]
            params = [str(s) for s in expr[1]]
            if var in params:
                return expr  # 束縛され直すので中へ入らない
            # 捕獲回避：val の自由変数とぶつかる引数名はリネーム
            fv = free_vars(val)
            new_params = []
            body = expr[2]
            changed = False
            for p in params:
                if p in fv:
                    p2 = fresh_var(fv | set(params) | free_vars(body))
                    body = alpha_rename(body, p, p2)
                    new_params.append(p2); changed = True
                else:
                    new_params.append(p)
            new_fn = [expr[0], [SYM(p) for p in new_params], body]
            return [new_fn[0], new_fn[1], substitute(body, var, val)] if changed \
                   else [expr[0], expr[1], substitute(body, var, val)]
        else:
            return [substitute(x, var, val) for x in expr]
    else:
        return expr

def desugar_do_setv_to_let(prog: Any) -> Any:
    """(do (setv x e) ... final) → (let [x e ...] final) へ（純粋 λ用）"""
    if not (is_list(prog) and prog and is_symbol(prog[0], "do")):
        return prog
    pairs = []
    *stmts, last = prog[1:]
    for s in stmts:
        if is_list(s) and len(s) == 3 and is_symbol(s[0], "setv") and isinstance(s[1], Symbol):
            pairs.extend([s[1], s[2]])
        else:
            # 非 setv を含む場合はそのまま do として残す（β評価では無視されうる）
            return prog
    return [SYM("let"), pairs, last]

def let_to_app(expr: Any) -> Any:
    """(let [x a y b ...] body) → ((fn [x y ...] body) a b ...)"""
    if not (is_list(expr) and expr and is_symbol(expr[0], "let")):
        return expr
#    items = vec_to_list(expr[1])
    items = expr[1]
    params, args = [], []
    for i in range(0, len(items), 2):
        params.append(items[i])
        args.append(items[i+1])
    lam = [SYM("fn"), params, expr[2]]
    return [lam, *args]

# 算術 / pow ,有限体上
def prim_eval_arith(op:str,expr: Any,numify:Any,mod:int=0)-> Any:
    args = [evaluate(x) for x in expr[1:]]
    if not all(is_number_atom(x) for x in args): return expr
    vals = [numify(x) for x in args]
    if(mod!=0):
        if op == "+": return sum(vals)
        if op == "-":
            return vals[0] if len(vals)==1 else vals[0] - sum(vals[1:])
        if op == "*":
            out = 1.0
            for v in vals: out *= v
            return out
        if op == "/":
            out = vals[0]
            for v in vals[1:]: out /= v
            return out
        if op == "pow":
            return vals[0] ** vals[1]
    else:
        if op == "+": return (sum(vals))%mod
        if op == "-":
            return vals[0]%mod if len(vals)==1 else (vals[0] - sum(vals[1:]))%mod
        if op == "*":
            out = 1.0
            for v in vals: out *= v
            return out%mod
        if op == "/":
            out = vals[0]
            for v in vals[1:]: out /= v
            return out%mod
        if op == "pow":
            return (vals[0] ** vals[1])%mod
            
def prim_eval(expr: Any) -> Any:
    """引数が全て数値等になったプリミティブを評価"""
    if not (is_list(expr) and expr and isinstance(expr[0], Symbol)):
        return expr
    op = str(expr[0])

    def numify(x): return float(x) if isinstance(x, int) else x

    # if
    if op == "if" and len(expr) == 4:
        cond = evaluate(expr[1])
        return evaluate(expr[2] if cond else expr[3])

    # 比較
    if op in {"<","<=",">",">=","=","!="} and len(expr) == 3:
        a = evaluate(expr[1]); b = evaluate(expr[2])
        if not (is_number_atom(a) and is_number_atom(b)): return expr
        a, b = numify(a), numify(b)
        return {
            "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "=": a == b, "!=": a != b
        }[op]

    # 算術 / pow
    if op in {"+","-","*","/","pow"}:
        return prim_eval_arith(op,expr,numify,MOD)

    # 単項/可変長
    if op in {"abs","round"} and len(expr) == 2:
        v = evaluate(expr[1])
        if not is_number_atom(v): return expr
        return abs(v) if op == "abs" else round(float(v))
    if op in {"max","min"} and len(expr) == 2 and is_list(expr[1]) and is_symbol(expr[1][0], "list"):
        arr = [evaluate(x) for x in expr[1][1:]]
        if not all(is_number_atom(x) for x in arr): return expr
        return max(arr) if op == "max" else min(arr)

    return expr  # それ以外は未評価のまま

def beta_step(expr: Any) -> Tuple[Any, bool]:
    """左最外 β 一歩 + let/do 脱糖 + プリミティブ簡約"""
    # let, do の脱糖
    if is_list(expr) and expr and is_symbol(expr[0], "do"):
        expr = desugar_do_setv_to_let(expr)
        if is_list(expr) and is_symbol(expr[0], "let"):
            expr = let_to_app(expr)
    if is_list(expr) and expr and is_symbol(expr[0], "let"):
        expr = let_to_app(expr)

    # ((fn [x] body) arg) → [x:=arg]body
    if is_list(expr) and expr:
        # 関数部を化簡
        head = expr[0]
        if is_list(head) and head and is_symbol(head[0], "fn"):
            # 複数引数なら左から1つずつ
            #params = [str(s) for s in vec_to_list(head[1])]
            params = [str(s) for s in head[1]]
            body = head[2]
            if not expr[1:]:
                return expr, False
            arg = expr[1]
            x = params[0]
            # 残りの引数がある場合は λ を1つ短くして再適用
            rest_params = params[1:]
            new_body = substitute(body, x, expr[1])
            if rest_params:
                new_fn = [SYM("fn"), [SYM(p) for p in rest_params], new_body]
                return [new_fn, *expr[2:]], True
            else:
                # 1引数λなら body へ置換して引数を1つ削る
                return (new_body if len(expr)==2 else [new_body, *expr[2:]]), True

    # 内部へ
    if is_list(expr):
        for i, x in enumerate(expr):
            nx, changed = beta_step(x)
            if changed:
                out = expr[:]
                out[i] = nx
                return out, True

    # プリミティブ（完全数値になっていれば評価）
    pe = prim_eval(expr)
    if pe is not expr:
        return pe, True
    return expr, False

def evaluate(expr: Any, max_steps: int = 10_000) -> Any:
    cur = expr
    for _ in range(max_steps):
        cur, changed = beta_step(cur)
        if not changed:
            break
    return cur

beta_eval_program_str =lambda  program_str:  evaluate(loads(program_str))
# -----------------------------
# 3) Dyck（括弧列）＋ラベル列（LIST/VEC 対応）
# -----------------------------
def sexp_to_dyck_and_labels(sexp: Any) -> Tuple[str, List[str]]:
    dyck, labels = [], []
    def visit(x: Any):
        dyck.append("(")
        if is_list(x):
            labels.append("LIST")
            for c in x: visit(c)
        elif isinstance(x, Symbol):
            labels.append(f"SYM:{str(x)}")
        elif isinstance(x, bool):
            labels.append(f"BOOL:{x}")
        elif is_number_atom(x):
            labels.append(f"NUM:{repr(x)}")
        elif x is None:
            labels.append("NONE")
        else:
            raise TypeError(f"Unsupported: {type(x)} {x}, {sexp}")
        dyck.append(")")
    visit(sexp)
    return "".join(dyck), labels

def dyck_and_labels_to_sexp(dyck: str, labels: List[str]) -> Any:
    dyck = re.sub(r"\s+", "", dyck)
    stack: List[list] = []
    root = None
    for ch in dyck:
        if ch == "(":
            stack.append([])
        elif ch == ")":
            node = stack.pop()
            if not stack:
                root = node
            else:
                stack[-1].append(node)
    if root is None: raise ValueError("bad Dyck")

    it = iter(labels)
    def build(n: list) -> Any:
        lab = next(it)
        if lab == "LIST":
            return [build(c) for c in n]
        if lab.startswith("SYM:"):
            if n: raise ValueError("SYM must be leaf")
            return SYM(lab[4:])
        if lab.startswith("NUM:"):
            if n: raise ValueError("NUM must be leaf")
            return ast.literal_eval(lab[4:])
        if lab.startswith("BOOL:"):
            if n: raise ValueError("BOOL must be leaf")
            return True if lab[5:]=="True" else False
        if lab == "NONE":
            if n: raise ValueError("NONE must be leaf")
            return None
        raise ValueError(f"unknown label {lab}")
    sexp = build(root)
    try:
        next(it)
        raise ValueError("labels too long")
    except StopIteration:
        pass
    return sexp

# -----------------------------
# 4) データセット作成
# -----------------------------
@dataclass
class ProgSample:
    sexp: str
    value_hy: Optional[Any]
    value_beta: Optional[Any]

isterminal=lambda v_hy:isinstance(v_hy, (int,float,bool)) and (not (isinstance(v_hy,float) and not math.isfinite(v_hy)))

def isOK(f):
    try:
        return f(),True
    except Exception:
        return None,False

def make_dataset(n: int, max_depth=4, max_bind=2, retries=100,fname="") -> List[ProgSample]:
    out: List[ProgSample] = []
    while len(out) < n:
        s=gen_program_with_setv_s(max_bind=max_bind, max_depth=max_depth)
        for _ in range(retries):
            v_hy,ok=isOK(lambda :hy_eval_program_str(s)) # ok=isterminal(v_hy) #式のままでいい
            # β-簡約（let などの糖衣脱ぎは beta_step 内で行われる）
            v_be,_=isOK(lambda :beta_eval_program_str(s))
            out.append(ProgSample(sexp=s, value_hy=float(v_hy) if isinstance(v_hy,(int,float)) else v_hy, value_beta=v_be))
    if(len(fname)>0):
        with open(fname,"w") as f:
            print(out,file=f)
    return out

def test(n: int, max_depth=4, max_bind=2,onlygen=False,debug=False,fname="") -> List[ProgSample]:
    out: List[ProgSample] = []
    for i in range(n):
        s=gen_program_with_setv_s(max_bind=max_bind, max_depth=max_depth)
        if(debug):
            print(s)
        if(not onlygen):
            v_hy = hy_eval_program_str(s)
            if(debug):
                print(v_hy)
            #print("isterminal",isterminal(v_hy))
            v_be = beta_eval_program_str(s)
            out.append(ProgSample(sexp=s, value_hy=float(v_hy) if isinstance(v_hy,(int,float)) else v_hy, value_beta=v_be))
    
    if(len(fname)>0):
        with open(fname,"w") as f:
            print(out,file=f)
    return out

# -----------------------------
# 5) デモ / CLI
# -----------------------------
def main(args):
    import time
    random.seed(0)
    print("Generating S-expressions...")
    t0 = time.time()
    ds = make_dataset(args.n, max_depth=args.max_depth, max_bind=args.max_bind)
    print(f"  generated: {len(ds)} samples in {time.time()-t0:.2f}s")

    for i, smp in enumerate(ds, 1):
        print(f"{i:02d}. {smp.sexp}  =>  HY:{smp.value_hy}  |  BETA:{smp.value_beta}")
        # Dyck 表示
        dyck, labs = sexp_to_dyck_and_labels(loads(smp.sexp))
        back = dumps(dyck_and_labels_to_sexp(dyck, labs))
        print("    Dyck:", dyck)
        print("    Labs:", labs[:8], "..." if len(labs)>8 else "")
        print("    Back:", back)
        print("-"*80)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_bind", type=int, default=2)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--onlygen", action="store_true")
    args = ap.parse_args()

    if(args.test):
        out=test(args.n,args.max_depth,args.max_bind,onlygen=args.onlygen)
        #print(out)
    else:
        main(args)        
    

