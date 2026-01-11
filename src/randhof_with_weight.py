import argparse        
import random
import math
# 値::=0,1,2,3,4,5,6
# bool::=True,False
# ops::=+,-,*,/
# cmps:==,<,>,<=,>=
# 変数::=x,y,z,t,v0,v1,...
# list::=[expr,...]
# expr::=値 | bool | 変数
#       | (ops expr,...）->　値
#       | (cmps expr,...）->　bool
#       | (if expr expr expr)-> expr
#       | (fn [変数,...] expr) -> closure
#       | (expr expr,...)-> expr
#       | (compose expr expr)-> expr
#       | (partial expr expr,...)-> expr
#       | (map expr list)-> list
#       | (filter expr list)-> list
#       | (reduce expr list　expr)-> expr
#       | (cons expr list)-> list
#       | (first list)-> expr
#       | (rest list)-> list
#       | (append list list)-> list
#       | (len list)-> 値

# 値・bool・変数・演算子など ------------------------------
MOD=7
VALUES = [str(i) for i in range(MOD)]          # 0..6
BOOLS = ["True", "False"]
BASE_VARS = ["o","p","q","r","s","t","u","v","x", "y", "z"]
OPS = ["+", "-", "*", "/"]
CMPS = ["==", "<", ">", "<=", ">="]

# kind はざっくり 4種類 + any
# "int", "bool", "list", "closure", "any"

def random_var(n_free_vars=5):
    # 変数 ::= x,y,z,t,v0,v1,...
    if random.random() < 0.5:
        return random.choice(BASE_VARS[:n_free_vars])
    else:
        return f"v{random.randint(0, 9)}"


# ---- 終端生成（深さが尽きたとき用） --------------------------
def gen_terminal(depth, want_kind="any",n_free_vars=5):
    # 深さ 0 のときに使う、できるだけ型を合わせた終端
    if want_kind == "int":# 値 or 変数を int 扱い
        if random.random() < 0.7:
            return random.choice(VALUES), "int"
        else:
            return random_var(), "int"
    elif want_kind == "bool":
        return random.choice(BOOLS), "bool"
    elif want_kind == "list":# 空リストを返しておく
        return "[]", "list"
    elif want_kind == "closure": # 単純な恒等関数
        v = random_var()
        return f"(fn [{v}] {v})", "closure"
    else:
        # any の場合はランダムに選ぶ
        choice = random.choice(["int", "bool", "list", "closure"])
        return gen_terminal(depth, choice)

# ---- list リテラル -----------------------------------------
def gen_list_literal(depth):    # list ::= [expr,...]
    # 深いほど要素数を増やし、各要素も複雑にする
    if depth <= 0:
        n = random.randint(0, 2)
    else:
        n = random.randint(2, 5)
    elems = [gen_expr(depth - 1, "any")[0] for _ in range(n)]
    return "[" + " ".join(elems) + "]", "list"

# ---- 各コンストラクタ（型つき） ----------------------------
def gen_op(depth):   # (opsymbols expr,... ) -> 値 (int)
    op = random.choice(OPS)
    arity = random.randint(2, 4)
    args = [gen_expr(depth - 1, "int")[0] for _ in range(arity)]
    return "(" + " ".join([op] + args) + ")", "int"

def gen_cmp(depth):  # (cmpsymbols expr,... ) -> bool
    op = random.choice(CMPS)
    arity = random.randint(2, 3)
    args = [gen_expr(depth - 1, "int")[0] for _ in range(arity)]
    return "(" + " ".join([op] + args) + ")", "bool"

def gen_if(depth, want_kind):  # (if expr expr expr) -> want_kind
    cond = gen_expr(depth - 1, "bool")[0]
    then_expr, _ = gen_expr(depth - 1, want_kind)
    else_expr, _ = gen_expr(depth - 1, want_kind)
    return f"(if {cond} {then_expr} {else_expr})", want_kind

def gen_fn(depth):    # (fn [変数,...] expr) -> closure
    num_params = random.randint(1, 3)
    params = [random_var() for _ in range(num_params)]
    # 関数本体は any 型にしておく
    body, _ = gen_expr(depth - 1, "any")
    return "(fn [" + " ".join(params) + "] " + body + ")", "closure"

def gen_app(depth, want_kind):    # (expr expr,...) -> expr (any)
    # 戻り値の静的型はわからないので any 扱い
    # ただし「計算ステップを増やす」ために、関数部は closure を優先
    fun_expr, _ = gen_expr(depth - 1, "closure")
    arity = random.randint(1, 3)
    args = [gen_expr(depth - 1, "any")[0] for _ in range(arity)]
    return "(" + " ".join([fun_expr] + args) + ")", "any"

def gen_compose(depth):    # (compose expr expr) -> closure とみなす
    f1= gen_expr(depth - 1, "closure")[0]
    f2= gen_expr(depth - 1, "closure")[0]   
    return f"(compose {f1} {f2})", "closure"

def gen_partial(depth):    # (partial expr expr,...) -> closure とみなす
    fun_expr, _ = gen_expr(depth - 1, "closure")
    arity = random.randint(1, 3)
    args = [gen_expr(depth - 1, "any")[0] for _ in range(arity)]
    pf=" ".join([fun_expr] + args)
    return f"(partial {pf})", "closure"

def gen_map(depth):    # (map expr list) -> list
    func_expr, _ = gen_expr(depth - 1, "closure")
    lst_expr, _ = gen_list(depth - 1)
    return f"(map {func_expr} {lst_expr})", "list"

def gen_filter(depth):    # (filter expr list) -> list
    pred_expr, _ = gen_expr(depth - 1, "closure")
    lst_expr, _ = gen_list(depth - 1)
    return f"(filter {pred_expr} {lst_expr})", "list"

def gen_reduce(depth, want_kind):    # (reduce expr list expr) -> expr (any / want_kind)
    func_expr, _ = gen_expr(depth - 1, "closure")
    lst_expr, _ = gen_list(depth - 1)
    init_expr, _ = gen_expr(depth - 1, want_kind)
    return f"(reduce {func_expr} {lst_expr} {init_expr})", want_kind

def gen_cons(depth):    # (cons expr list) -> list
    e_expr, _ = gen_expr(depth - 1, "any")
    lst_expr, _ = gen_list(depth - 1)
    return f"(cons {e_expr} {lst_expr})", "list"

def gen_first(depth):    # (first list) -> expr (any)
    lst_expr, _ = gen_list(depth - 1)
    return f"(first {lst_expr})", "any"

def gen_rest(depth):    # (rest list) -> list
    lst_expr, _ = gen_list(depth - 1)
    return f"(rest {lst_expr})", "list"

def gen_append(depth):    # (append list list) -> list
    lst1, _ = gen_list(depth - 1)
    lst2, _ = gen_list(depth - 1)
    return f"(append {lst1} {lst2})", "list"

def gen_len(depth):    # (len list) -> 値 (int)
    lst_expr, _ = gen_list(depth - 1)
    return f"(len {lst_expr})", "int"

def gen_let(depth): #(let bindings body) -> expr (any)
    names = [random_var() for _ in range(3)]
    binds = []
    for n in names:
        rhs,_ = gen_expr(depth-1,"any")
        binds.append(f"({n} {rhs})")
    body,_ = gen_expr(depth-1,"any")
    return f"(let ({' '.join(binds)}) {body})", "any"

# list 生成用ヘルパー（list 型を返す式全般）
def gen_list(depth):
    if depth <= 0:
        return gen_list_literal(depth)

    # list を返す候補たち
    candidates = [
        ("literal", 2),
        ("map", 4),
        ("filter", 3),
        ("cons", 3),
        ("rest", 2),
        ("append", 3),
    ]
    kinds = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]
    kind = random.choices(kinds, weights=weights, k=1)[0]

    kinds={"literal":gen_list_literal,
           "map":gen_map,
           "filter":gen_filter,
           "cons":gen_cons,
           "rest":gen_rest,
           "append":gen_append,
           }
    return kinds.get(kind,gen_list_literal)(depth)

# ---- expr 生成のメイン -------------------------------------
def gen_expr(depth, want_kind="any",n_free_vars=4):
    """
    指定された戻り値「kind」を持つ expr をランダム生成する。
    want_kind: "int", "bool", "list", "closure", "any"
    """
    if depth <= 0:
        return gen_terminal(depth, want_kind,n_free_vars)

    # 深さがまだある場合は、なるべく「計算ステップが多くなりそうな」コンストラクタを優先
    # kindごとに候補を用意
    kind_ccanditates={
        "int":[
             ("value_terminal", 1),
             ("op", 4),
             ("len", 3),
             ("reduce", 4),
             ("if_int", 2),
             ("app", 2),
        ],
        "bool":[
             ("bool_terminal", 1),
             ("cmp", 5),
             ("if_bool", 2),
             ("app", 1),
        ],
        "list":[
             ("list", 5),
             ("if_list", 2),
             ("app", 1),
        ],
        "closure":[
             ("fn", 4),
             ("compose", 3),
             ("partial", 3),
             ("if_closure", 1),
        ],
        "withlet":[
             ("value_terminal", 4),
             ("op", 6),
             ("cmp", 3),
             ("if_int", 3),
             ("let", 2),
             ("app", 2),
        ],
        "kinder":[
             ("value_terminal", 4),
             ("op", 6),
             ("cmp", 3),
             ("if_int", 3),
             ("let", 2),
             ("app", 2),
        ],
        "default":[
             ("op", 2),
             ("cmp", 1),
             ("list", 2),
             ("fn", 2),
             ("let", 2),
             ("app", 4),
             ("compose", 2),
             ("partial", 2),
             ("if_any", 2),
             ("first", 2),
             ("reduce", 3),
         ]
    }
    candidates=kind_ccanditates.get(want_kind,kind_ccanditates["default"])
    kinds = [c[0] for c in candidates]
    weights = [c[1] for c in candidates]
    k = random.choices(kinds, weights=weights, k=1)[0]
    cases={
        "value_terminal":lambda d:gen_terminal(d, "int"),
        "bool_terminal":lambda d:gen_terminal(d, "bool"),
        "op":gen_op,
        "cmp":gen_cmp,
        "len":gen_len,
        "fn":gen_fn,
        "compose":gen_compose,
        "partial":gen_partial,
        "list":gen_list,
        "map":gen_map,
        "filter":gen_filter,
        "reduce":lambda d: gen_reduce(d, "any" if want_kind == "any" else want_kind),
        "first":gen_first,
        "app":lambda d:gen_app(d, want_kind),
        # if 系は want_kind に応じて
        "if_int": lambda d:gen_if(d, "int"),
        "if_bool":lambda d:gen_if(d, "bool"),
        "if_list":lambda d:gen_if(d, "list"),
        "if_closure":lambda d:gen_if(d, "closure"),
        "if_any":lambda d:gen_if(d, "any"),
        }
    return cases.get(k,lambda d:gen_terminal(d, want_kind))(depth)
 
  
# ---- 外から呼ぶ用のラッパー -------------------------------
def random_typed_sexp(max_depth=5, want_kind="any", seed=None,n_free_vars=5):
    """
    型と戻り値を意識したランダム S 式生成。
    want_kind: "int", "bool", "list", "closure", "any"
    """
    if seed is not None:
        random.seed(seed)
    expr, kind = gen_expr(max_depth, want_kind,n_free_vars=n_free_vars)
    return expr, kind

def random_typed_sexp_n(n,max_depth=5,want_kind="int", seed=None):
    # 例: int を返す複雑な式をたくさん生成
    for i in range(n):
        e, k = random_typed_sexp(max_depth=max_depth, want_kind=want_kind)
        #print(k, "=>", e)

######
import re
import random
from typing import Any, Dict, Tuple

# ---------- Tokenizer & Parser ----------

TOKEN_REGEX = re.compile(r"""
    \s*(
        \d+ |              # numbers
        True|False|        # booleans
        <=|>=|== |         # multi-char operators
        [()\[\]] |         # parens and brackets
        [+\-*/<>] |        # single-char ops
        [A-Za-z_][A-Za-z0-9_]*  # identifiers
    )
""", re.VERBOSE)


def tokenize(s: str):
    tokens = TOKEN_REGEX.findall(s)
    return tokens

def parse(tokens):
    """Lisp 風 S 式 + [ ... ] リストを AST に変換"""
    pos = 0

    def parse_expr():
        nonlocal pos
        if pos >= len(tokens):
            raise SyntaxError("Unexpected EOF")
        tok = tokens[pos]
        pos += 1

        if tok == "(":
            lst = []
            while pos < len(tokens) and tokens[pos] != ")":
                lst.append(parse_expr())
            if pos >= len(tokens) or tokens[pos] != ")":
                raise SyntaxError("Missing )")
            pos += 1
            return lst

        if tok == "[":
            elems = []
            while pos < len(tokens) and tokens[pos] != "]":
                elems.append(parse_expr())
            if pos >= len(tokens) or tokens[pos] != "]":
                raise SyntaxError("Missing ]")
            pos += 1
            return ("list", elems)

        # atom
        if tok.isdigit():
            return int(tok)
        if tok == "True":
            return True
        if tok == "False":
            return False
        return tok  # symbol / variable name

    expr = parse_expr()
    if pos != len(tokens):
        raise SyntaxError("Extra tokens after expression")
    return expr

# ---------- Evaluator ----------
def is_closure(v: Any) -> bool:
    return isinstance(v, dict) and v.get("type") == "closure"

def copy_env(env: Dict[str, Any]) -> Dict[str, Any]:
    return dict(env)

def make_new_env(env,params,arg_vals):
    new_env = copy_env(env)
    for p, v in zip(params, arg_vals):
        new_env[p] = v
    return new_env

def eval_expr(expr, env=None) -> Tuple[Any, int]:
    """
    expr を評価し、(結果, 簡約ステップ数) を返す
    """
    if env is None:
        env = {}
    return _eval(expr, env)

def _eval(expr, env):
    steps = 0
    if isinstance(expr, (int, bool)):# --- atoms ---
        return expr, 0
    elif isinstance(expr, str): # 変数なら環境から
        #print("環境から変数へ")
        if expr in env:
            return env[expr], 0
        # 未束縛変数 → これ以上簡約しない
        return expr, 0
    # --- list literal / list value ---
    elif isinstance(expr, tuple) and expr and expr[0] == "list":
        elems ,steps= _evalargs(expr[1],env,steps)
        return ("list", elems), steps
    # --- S 式 (list) ---
    elif isinstance(expr, list):
        if not expr:
            return expr, 0
        head = expr[0]
        # 特別扱いの構文
        if isinstance(head, str):
            if head in OPS:
                 return _eval_op(head, expr[1:], env)
            elif head in CMPS:
                 return _eval_cmp(head, expr[1:], env)
            else:
                specialfuncs={
                    "if":_eval_if,
                    "fn":_eval_fn,
                    "let":_eval_let,
                    "compose":_eval_compose,
                    "partial": _eval_partial,
                    "map":_eval_map,
                    "filter":_eval_filter,
                    "reduce":_eval_reduce,
                    "cons":_eval_cons,
                    "first":_eval_first,
                    "rest":_eval_rest,
                    "append": _eval_append,
                    "len":_eval_len,
                }
            return specialfuncs.get(head,_eval_app)(expr, env)
        else:# それ以外は「関数適用」
            return _eval_app(expr,env)
    # その他はそのまま
    return expr, 0

def _evalarg(arg,env,steps):
    v, st = _eval(arg, env)
    steps += st
    return v,steps

def _evalargs(args,env,steps): 
    if(isinstance(args,list)):
        vals = []
        for a in args:
            v, st = _eval(a, env)
            steps += st
            vals.append(v)
        return vals,steps
    else:
        return _evalarg(args,env,steps)

def _evalarg2(args1,args2,env,steps=0): 
    v1,steps=_evalarg(args1,env,steps)
    v2,steps=_evalarg(args2,env,steps)
    return v1,v2,steps

def _evalargs2(args1,args2,env,steps): 
    v1,steps=_evalargs(args1,env,steps)
    v2,steps=_evalargs(args2,env,steps)
    return v1,v2,steps

def check_list0(lst_v):
    return isinstance(lst_v, tuple) and lst_v[0] == "list" 

def check_list(lst_v):
    return check_list0(lst_v) and lst_v[1]

def _valid(check,answer,invalid,step):
    if(check):
        return answer,step+1
    else:
        return invalid, step

   
# --- 個別の構文の評価関数 ---
def _eval_if(expr, env):    # (if cond then else)
    if len(expr) != 4:
        return expr, 0
    _, cond_e, then_e, else_e = expr
    cond_v,steps=_evalarg(cond_e,env,0)
    if isinstance(cond_v, bool):
        steps += 1  # ブランチ選択自体を 1 ステップと数える
        target = then_e if cond_v else else_e
        v, st2 = _eval(target, env)
        steps += st2
        return v, steps
    # 条件が bool でなければ部分簡約のまま返す
    return ["if", cond_v, then_e, else_e], steps

def _eval_fn(expr, env):    # (fn [params...] body) -> closure
    if len(expr) != 3:
        return expr, 0
    _, params_e, body = expr

    # [x y z] は parser では ("list", ["x","y","z"]) になっている
    if isinstance(params_e, tuple) and params_e[0] == "list":
        raw = params_e[1]
    else:
        raw = params_e

    if not (isinstance(raw, list) and all(isinstance(p, str) for p in raw)):
        return expr, 0
    else:
        closure = {
            "type": "closure",
            "params": list(raw),
            "body": body,
            "env": copy_env(env),
        }
        # クロージャ構築を 1 ステップとカウント
        #print("closure",closure)
        return closure, 1

def _eval_let(expr, env):
    """
    (let ((x e1) (y e2) ...) body)
    - e1,e2,... は「外側 env」で評価（並列束縛）
    - body は拡張 env で評価
    """
    if len(expr) != 3:
        return expr, 0  # あるいは例外
    _, bindings, body = expr

    if not isinstance(bindings, list):
        return expr, 0
    else:  # (1) RHS を外側 env で評価してからまとめて束縛（並列）
        evaluated = []
        for b in bindings:
            if not (isinstance(b, list) and len(b) == 2 and isinstance(b[0], str)):
                return expr, 0
            name, rhs = b[0], b[1]
            v,steps=_evalarg(rhs,env,0)
            evaluated.append((name, v))

        # (2) env を拡張（外側を壊さない）
        new_env=make_new_env(env,*evaluated)
        # (3) body を new_env で評価
        return _evalarg(body,new_env,steps)

def _eval_op(op, args, env):
    vals,steps=_evalargs(args,env,0)

    if all(isinstance(v, int) for v in vals) and vals:
        steps += 1
        if op == "+":
            return sum(vals)%MOD, steps
        elif op == "-":
            if len(vals) == 1:
                return -vals[0]%MOD, steps
            elif(len(vals) == 2):
                return (vals[0]-vals[1])%MOD ,steps            
            else:
                return (vals[0]-sum(vals[1:]))%MOD ,steps
        elif op == "*":
            return math.prod(vals)%MOD, steps
        elif op == "/":
            res = vals[0]
            for x in vals[1:]:
                if x == 0:# 0 で割れない → これ以上簡約しない
                    return [op] + vals, steps
                res = res / x
            return res%MOD, steps
    # 完全には簡約できない場合は部分式として残す
    return [op] + vals, steps

def _eval_cmp(op, args, env):
    vals,steps=_evalargs(args,env,0)
    if all(isinstance(v, (int, bool)) for v in vals) and len(vals) >= 2:
        steps += 1
        cands={"==":lambda vals:all(vals[i] == vals[i+1] for i in range(len(vals) - 1)),
               "<": lambda vals:all(vals[i] < vals[i+1] for i in range(len(vals) - 1)),
               ">": lambda vals:all(vals[i] > vals[i+1] for i in range(len(vals) - 1)),
               "<=": lambda vals:all(vals[i] <= vals[i+1] for i in range(len(vals) - 1)),
               ">=": lambda vals:all(vals[i] >= vals[i+1] for i in range(len(vals) - 1)),
            }
        return cands.get(op,lambda vals:False)(vals), steps
    return [op] + vals, steps

def _eval_app(expr, env):  # (f a1 a2 ...)
    f_val   ,steps=_evalarg(expr[0],env,0) #head f
    arg_vals,steps =_evalarg(expr[1:],env,0) #args

    if is_closure(f_val):#fが関数
        params = f_val["params"]
        body = f_val["body"]
        if isinstance(params,list) or len(arg_vals) != len(params):
            # 引数個数が合わない場合は簡約しない
            if(isinstance(arg_vals,list)):
                return [f_val] + arg_vals, steps
            else:
                return [f_val, arg_vals], steps
        else:
            #fの評価結果をenvに追加
            new_env=make_new_env(f_val["env"],params,arg_vals)
            # β 簡約
            return _evalargs(body,new_env,steps+1)
    else:
        # 関数でなければこれ以上簡約できない
        if(isinstance(arg_vals,list)):
            return [f_val] + arg_vals, steps
        else:
            return [f_val, arg_vals], steps
        
def _eval_compose(expr, env): # (compose f g)  ~> (fn [x] (f (g x)))
    if len(expr) != 3:
        return expr, 0
    _, f_e, g_e = expr
    param = "_c_arg"
    composed_ast = ["fn", [param], [f_e, [g_e, param]]]
    v, st = _eval(composed_ast, env)
    return v, st + 1  # compose 自体を 1 ステップとカウント

def _eval_partial(expr, env):   # (partial f a1 a2 ...)
    if len(expr) < 3:
        return expr, 0
    _, f_e, *arg_es = expr
    f_val,steps=_evalarg(f_e,env,0)
    arg_vals,steps=_evalarg(arg_es,env,steps)

    if not is_closure(f_val):
        return ["partial", f_val] + arg_vals, steps

    params = f_val["params"]
    body = f_val["body"]

    # 引数が十分ならその場で適用
    if len(arg_vals) >= len(params):
        return _evalarg(body, make_new_env(f_val["env"],params,arg_vals),steps+1)
    else:
        # そうでなければ、残りの引数をとるクロージャを返す
        remaining_params = params[len(arg_vals) :]
        closure = {
            "type": "closure",
            "params": remaining_params,
            "body": body,
            "env": make_new_env(f_val["env"],params,arg_vals),
        }
        return closure, steps+1

def _eval_map(expr, env):    # (map f list) -> list
    if len(expr) != 3:
        return expr, 0
    _, f_e, list_e = expr
    f_val,steps,=_evalarg(f_e,env,0)
    lst_v,steps,=_evalarg(list_e,env,steps)

    #fが関数でないか残りの引数がlist出ない場合はそのまま返す
    if not is_closure(f_val) or not (isinstance(lst_v, tuple) and lst_v[0] == "list"):
        return ["map", f_val, lst_v], steps
    else:
        result_elems = []
        for elem in lst_v[1]:
            applied, st3 = _eval_app([f_val, elem], env)
            steps += st3
            result_elems.append(applied)

        return ("list", result_elems), steps+1

def _eval_filter(expr, env):    # (filter pred list) -> list
    if len(expr) != 3:
        return expr, 0
    _, pred_e, list_e = expr
    p_val,steps,=_evalarg(pred_e,env,0)
    lst_v,steps,=_evalarg(list_e,env,steps)

    if not is_closure(p_val) or not (isinstance(lst_v, tuple) and lst_v[0] == "list"):
        return ["filter", p_val, lst_v], steps
    else:
        result = []
        for elem in lst_v[1]:
            cond, st3 = _eval_app([p_val, elem], env)
            steps += st3
            if isinstance(cond, bool):
                if cond:
                    result.append(elem)
            else:
                # 判定できない場合は残しておく
                result.append(elem)

        return ("list", result), steps+1

def _eval_reduce(expr, env):  # (reduce f list init) -> expr
    if len(expr) != 4:
        return expr, 0
    _, f_e, list_e, init_e = expr
    f_val, steps = _evalarg(f_e, env,0)
    lst_v, steps = _evalarg(list_e, env,steps)
    acc, steps = _evalarg(init_e, env,steps)

    if not is_closure(f_val) or not (isinstance(lst_v, tuple) and lst_v[0] == "list"):
        return ["reduce", f_val, lst_v, acc], steps
    else:
        for elem in lst_v[1]:
            acc, st4 = _eval_app([f_val, acc, elem], env)
            steps += st4
        return acc, steps+1

def _eval_cons(expr, env):    # (cons expr list) -> list
    if len(expr) != 3:
        return expr, 0
    _, head_e, list_e = expr
    h_v, steps = _evalarg(head_e, env,0)
    lst_v, steps = _evalarg(list_e, env,steps)

    if isinstance(lst_v, tuple) and lst_v[0] == "list":
        return ("list", [h_v] + lst_v[1]), steps+1
    else:
        return ["cons", h_v, lst_v], steps

def _eval_first(expr, env):    # (first list) -> expr
    if len(expr) != 2:
        return expr, 0
    _, list_e = expr
    lst_v, steps = _evalarg(list_e, env,0)

    if isinstance(lst_v, tuple) and lst_v[0] == "list" and lst_v[1]:
        return lst_v[1][0], steps+1
    else:
        return ["first", lst_v], steps

def _eval_rest(expr, env):  # (rest list) -> list
    if len(expr) != 2:
        return expr, 0
    _, list_e = expr
    lst_v, steps = _evalarg(list_e, env,0)

    if isinstance(lst_v, tuple) and lst_v[0] == "list" and lst_v[1]:
        return ("list", lst_v[1][1:]), steps+1
    else:
        return ["rest", lst_v], steps

def _eval_append(expr, env):    # (append list list) -> list
    if len(expr) != 3:
        return expr, 0
    _, l1_e, l2_e = expr
    l1_v, steps = _evalarg(l1_e, env,0)
    l2_v, steps = _evalarg(l2_e, env,steps)

    if (isinstance(l1_v, tuple)     and l1_v[0] == "list"
        and isinstance(l2_v, tuple) and l2_v[0] == "list" ):
        return ("list", l1_v[1] + l2_v[1]), steps+1
    else:
        return ["append", l1_v, l2_v], steps

def _eval_len(expr, env):    # (len list) -> 値
    if len(expr) != 2:
        return expr, 0
    _, list_e = expr
    lst_v, steps = _evalargs(list_e, env,0)

    if isinstance(lst_v, list) and lst_v[0] == "list":
        return len(lst_v[1]), steps+1
    else:
        return ["len", lst_v], steps

### to output ###
def sexpr_to_str(expr) -> str:
    """内部表現を S 式文字列に戻す"""
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, bool):
        return "True" if expr else "False"
    if isinstance(expr, str):
        return expr
    if isinstance(expr, dict) and expr.get("type") == "closure":
        params = " ".join(expr["params"])
        body = sexpr_to_str(expr["body"])
        return f"(closure fn [{params}] {body})"
    if isinstance(expr, tuple) and expr and expr[0] == "list":
        inner = " ".join(sexpr_to_str(e) for e in expr[1])
        return f"[{inner}]"
    if isinstance(expr, list):
        inner = " ".join(sexpr_to_str(e) for e in expr)
        return f"({inner})"
    return repr(expr)

def totaleval(expr_str: str):
    return eval_expr(parse(tokenize(expr_str)))

def reduce_and_show(expr_str: str):
    """
    文字列で与えられた S 式を:
      - パース
      - 評価 (できる限り簡約)
      - 結果とステップ数を print するヘルパー
    """
    tokens = tokenize(expr_str)
    ast = parse(tokens)
    value, steps = eval_expr(ast)
    print("元の式:   ", expr_str)
    print("AST:      ", sexpr_to_str(ast))
    print("簡約結果: ", sexpr_to_str(value))
    print("ステップ数:", steps)
    print("-" * 60)

def eval_demo():  # 簡単なデモ
    samples = [
        "(+ 1 2 3)",
        "(if True (+ 1 2) 5)",
        "((fn [x] (+ x 1)) 3)",
        "(map (fn [x] (+ x 1)) [1 2 3])",
        "(filter (fn [x] (> x 2)) [1 2 3 4])",
        "(reduce (fn [a b] (+ a b)) [1 2 3] 0)",
        "(len [1 2 3 4])",
        "(compose (fn [x] (+ x 1)) (fn [y] (* y 2)))",
    ]
    samples=[
        "(len (1 2 3 4))",
        "(len [1 2 3 4])",
        "(len (list [1 2 3 4]))",
        "((fn [x] (x+1)) 2)",
        "((fn [x,f] (if (== x 0 ) 0 (f x-1))) 4)",
        "((let x 4 y 5 (+ x y)))"
    ]
    print("demo")
    for s in samples:
        reduce_and_show(s)

def gen_and_eval(num_exprs=5, max_depth=4, want_kind="int",n_free_vars=4, seed=None):
    if seed is not None:
        random.seed(seed)
    result=[]
    for _ in range(num_exprs):
        expr_str, kind = random_typed_sexp(max_depth=max_depth, want_kind=want_kind,n_free_vars=n_free_vars)
        value, steps=totaleval(expr_str)
        result.append((expr_str,sexpr_to_str(value),steps))
    return result

def gen_and_eval_print(params,filename="sexppair"):
    seed=params.seed
    show=params.show
    num_exprs=params.n
    max_depth=params.max_depth
    target_free=params.target_free
    valtype=params.valtype
    if seed is not None:
        random.seed(params.seed)
    with open(f"{filename}_n{num_exprs}_d{max_depth}_freevar{target_free}_kind{valtype}.txt", "w") as f:
        for i in range(num_exprs):
            expr_str, _= random_typed_sexp(max_depth=params.max_depth, want_kind=params.valtype)
            value, steps=totaleval(expr_str)
            if(show):
                print(i,"-" * 40 )
                print("Generated expr:", expr_str)
                print("Evaluated to   :", sexpr_to_str(value))
                print("Steps taken    :", steps)
            else:
                print(f"{expr_str}, {sexpr_to_str(value)}, {steps}", file=f)
import argparse
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random higher‑order S‑expr generator + partial evaluator (Python + hy models, with step + free‑var counting + list forms)")
    p.add_argument("--n", type=int, default=10, help="number of expressions")
    p.add_argument("--max_depth", type=int, default=4, help="maximum nesting depth")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--closed", action="store_true", help="disallow free variables in generation (forces 0 free vars)")
    p.add_argument("--var-pool", type=int, default=4, help="size of the free-variable symbol pool (x,y,z,t,v0,...) ")
    p.add_argument("--target_free", type=int, default=None, help="require exactly this many distinct free variables (best-effort)")
    p.add_argument("--show",  action="store_true", help="show result")
    p.add_argument("--show_short",  action="store_true", help="show short result")
    p.add_argument("--valtype",  type=str, default="int", help="value type to generate (int, bool, etc.)")
    p.add_argument("--demo",  action="store_true", help="show demo")
    return p.parse_args()

if __name__ == "__main__":
    a=parse_args()
    if(a.demo):
        eval_demo()
    else:
        S=gen_and_eval(a.n,a.max_depth,seed=42)
        print("length,")
        for s in S:
            print(s[0])
        for s in S:
            print(len(s[0]),len(s[1]),s[2],"steps")
    
    #ns = parse_args()
    #gen_and_eval_print(ns)