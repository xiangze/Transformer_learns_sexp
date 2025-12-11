import argparse
import re
from difflib import context_diff
from collections import Counter, OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple

#()->[]
listhreads=["("+" "*i+"list" for i in range(10)]
#括弧の数だけを数える
def _replace_list(inss:str,debug=False,addlist=True):
    depth=0
    kakkostack=[]#[/]のdepth
    listfound=False
    tmp=""
    ins=inss
    while(ins!=""):
         listfound=False
         if(debug):
             print(f"s={ins[0]}, depth:{depth} input:{ins}  out:{tmp} stack {kakkostack}")
         assert(depth>=0)
         for ls in listhreads: #"( list"
            l=len(ls)
            if(len(ins)>l and ins[:l]==ls):    
                kakkostack.append(depth)
                depth+=1
                tmp="".join([tmp]+[ls]*addlist+[" ["])
                ins=ins[l:]
                listfound=True
                break
         if(not listfound):
            s=ins[0]
            if(s=="("):
                depth+=1
                tmp+="("
            elif(s==")"):
                depth-=1
                if(depth>0 and len(kakkostack)>0 and depth==kakkostack[-1]):
                    kakkostack.pop()
                    if(addlist):
                        tmp+="])"
                    else:
                        tmp+="]"
                else:
                    tmp+=")"
            else:
                tmp+=s
            ins=ins[1:]
    return tmp

def kakkocheck(ins:str):
    depth=0
    while(ins!=""):
        s=ins[0]
        if("(" in s): depth+=1
        if(s==")"):  depth-=1
        ins=ins[1:]
    assert(depth==0)    
    return depth==0

def showdepth(inss:str):
    out=""
    depth=0
    ins=inss.replace("("," (").replace(")"," ) ").replace("  "," ").split(" ")
    while(ins!=[]):
        s=ins[0]
        if("(" in s):
            depth+=1
        out+=" "*depth+s+":"+str(depth)
        if(s==")"):
            depth-=1
        ins=ins[1:]
    return out

replace_list=lambda s,debug=False,n=10,addlist=True:_replace_list(s,debug=debug,addlist=addlist)


def label_brackets(expr: str) -> str:
    """
    外側から内側へ向かう通し番号で (), [] のペアに番号を付ける。
    開き括弧で番号を新規発番→スタックに push、対応する閉じ括弧で pop して同じ番号を付与。
    表現:  '(' -> '(#k',  ')' -> ')#k',  '[' -> '[#k',  ']' -> ']#k'
    """
    stack = []  # list of (bracket_type, id)
    next_id = 1
    out_chars = []

    for ch in expr:
        if ch == '(' or ch == '[':
            this_id = next_id
            next_id += 1
            stack.append((ch, this_id))
            out_chars.append(f"{ch}#{this_id}")
        elif ch == ')' or ch == ']':
            if not stack:
                raise ValueError(f"Unmatched closing bracket: {ch}")
            open_ch, open_id = stack.pop()
            if (open_ch == '(' and ch != ')') or (open_ch == '[' and ch != ']'):
                raise ValueError(f"Mismatched brackets: opened {open_ch} but closed {ch}")
            #out_chars.append(f"{ch}#{open_id}")
            out_chars.append(f"#{ch}{open_id}")
        else:
            out_chars.append(ch)
    if stack:
        # 未クローズの括弧が残っている
        raise ValueError(f"Unmatched opening bracket(s): {stack}")
    return "".join(out_chars)

#S式の括弧()と角括弧[]に外側から番号を付けた上でそれを含めたスペースで区切られた記号の種類を数え、それぞれに番号をつけるp
def count_and_index_tokens(labeled_expr: str):
    """
    空白で分割したトークン列の頻度カウントと、出現順に基づくトークン種のID付与を行う。
    """
    tokens = labeled_expr.split()
    counts = Counter(tokens)
    # 出現順で安定に ID を割り当てる
    seen = OrderedDict()
    for tok in tokens:
        if tok not in seen:
            seen[tok] = len(seen)   # 0 始まり
    return tokens, counts, dict(seen)

def analyze_s_expr(expr: str):
    """
    まとめ：番号付き文字列、トークン数、種類数、各トークン種のIDと頻度を返す。
    """
    labeled = label_brackets(expr)
    tokens, counts, token2id = count_and_index_tokens(labeled)
    return {
        "labeled_expr": labeled,
        "num_tokens": len(tokens),
        "num_unique_tokens": len(counts),
        "token_index": token2id,     # {トークン: 出現順ID}
        "token_counts": counts,      # Counter
        }

#def sexp_str_to_dyck_and_labels(sexp_str: str) -> Tuple[str, List[str]]:
def sexp2ilist(tokens:List[str])->List[str]:
    for tok in tokens:
        print(tok)
    #id_to_onehot(tokens)
def sexps2idicks(tokens:List[str]) -> List[Tuple[str, List[str]]]:
    [t for t in tokens]
    pass

def test_kakkoid( expr = '(do (setv a ( y (+ s b ) ) ))'  ,withsorted=False):
    #expr = '(do (setv y (sum (list  (min (list  ((fn [v] 3) -4.77) (if (> 8.79 x) -7 t) (pow y 4) x)) (* ((fn [a] 3) 4) (* 4.56 -3)) ((fn [v] (+ -8 v)) (+ 6.8 0.97)) ((fn [c] (pow t -1)) (if (= -3 a) -1 1.06))))))'
    kakkocheck(expr)
    result = analyze_s_expr(expr)

    print("=== Labeled Expression ===")
    print(result["labeled_expr"])
    print("=== Summary ===")
    print(f"Total tokens:  {result['num_tokens']}")
    print(f"Unique tokens: {result['num_unique_tokens']}")
    print("=== Token Types (ID -> token : count) ===")
    for  tok, i in result["token_index"].items():
        print(tok)
    if(withsorted):
        print("---sorted---")
        # 出現順 ID で並べ替えて見やすく表示
        inv = sorted(((i, tok) for tok, i in result["token_index"].items()), key=lambda x: x[0])
        for i, tok in inv:
            print(f"{i:>3} : {tok} : {result['token_counts'][tok]}")

def testlist(pat=[],debug=False):
    patterns=[
        ["(do (list -6.72 (list (* -4 9) 4 x) ((fn [y] -9) -8.38) v))",
         "(do (list [ -6.72 (list [ (* -4 9) 4 x]) ((fn [y] -9) -8.38) v]) )"],
         ["(do (list a (list x y z) e f) d )",
          "(do (list [a (list [x y z]) e f]) d )"         ],
          ["(do (list a (list x y) e f) d)",
           "(do (list [a (list [x y]) e f]) d)"           ],
           ["(do (list a (list x (list y z)) e f) d )",
            "(do (list [a (list [x (list [y z]) ]) e f]) d )"             ],
        ["(do (setv y (sum (list  (min (list  ((fn [v] 3) -4.77) (if (> 8.79 x) -7 t) (pow y 4) x)) (* ((fn [a] 3) 4) (* 4.56 -3)) ((fn [v] (+ -8 v)) (+ 6.8 0.97)) ((fn [c] (pow t -1)) (if (= -3 a) -1 1.06))))))",
         "(do (setv y  (sum  (list[   (min (list[   ((fn [v] 3) -4.77) (if (> 8.79 x) -7 t) (pow y 4) x   ])  )  (* ((fn [a] 3) 4) (* 4.56 -3))  ((fn [v] (+ -8 v)) (+ 6.8 0.97))  ((fn [c] (pow t -1)) (if (= -3 a) -1 1.06))  ]) ) ))"
        ],
    ]
    if(pat!=[]):
        patterns=pat
        
    for i,p in enumerate(patterns):
        org,exp=p
        if(not kakkocheck(org)):
            print("kakko not match")
            exit()
        try:
            s=replace_list(org,n=5,debug=False).replace("[ ","[")
            if(exp.replace(" ","")!=s.replace(" ","")):
                print(f"differ {i}th !!")
                print(f"in {len(org)}:  {org}")
                print("out:",s)
                print("exp:",exp)
                faildiff(org,exp)
                exit()      
            else:
                print(f"success {i}")      
        except Exception as e:
            print(f"exception at {i}th",e)
            faildiff(org,exp)
            exit()

def faildiff(org,exp):
    out=replace_list(org,n=5,debug=True).replace("[ ","[")
    out=showdepth(out)
    exp=showdepth(exp)
    print(context_diff(out,exp))

def testlist_regression(n,max_bind, max_depth,debug=False): 
    import generate_sexp_with_variable as g
    for i in range(n):
        s=g.replace_sexp(g.gen_program_with_setv(max_bind, max_depth))
        testlist([s],debug)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_bind", type=int, default=2)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--addid", action="store_true")
    args = ap.parse_args()
    if(args.regression):
        testlist_regression(args.n,args.max_depth,args.max_bind)
    elif(args.addid):
        test_kakkoid()
    else:
        testlist()