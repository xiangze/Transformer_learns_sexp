
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

def countparen(s: str) -> int:
    """S式文字列 s に含まれる括弧の深さをカウント"""
    max_depth = 0
    current_depth = 0
    for char in s:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    return max_depth
def countkakko(s: str) -> int:  
    """S式文字列 s に含まれる[]の深さをカウント"""
    max_depth = 0
    current_depth = 0
    for char in s:
        if char == '[':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ']':
            current_depth -= 1
    return max_depth
    
def sexp_str_to_dyck_and_labels(S:str,show=False) -> List[int]:
     depth=0
     kdepth=0
     Dyks = []
     maxdepth=countparen(S)
     """S式文字列 s に含まれる単語数をカウント"""
     contents = S.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').split()
     worddict={token:i for i, token in enumerate(list(set(contents))) }
     tokens = S.replace('(', '( ').replace(')', ' )').replace('[', '[ ').replace(']', ' ]').split()

     if(show):
        print(contents)
        print(worddict)
        print(tokens)
        
     for s in tokens:
        if(s=='('):
            depth+=1
            Dyks.append(depth*2)
        elif(s==')'):
            Dyks.append(depth*2+1)
            depth-=1
        elif(s=='['):
            kdepth+=1
            Dyks.append(-kdepth*2)
        elif(s==']'):
            Dyks.append(-kdepth*2-1)
            kdepth-=1
        else:
            Dyks.append(worddict[s]+maxdepth)
     Dyks.append(0) # End token
     return Dyks

if __name__ == "__main__":
    examples = [
        "(+ 1 (* 2 3))",
        "(if (< 1 2) (pow 3 2) (min (list 7 8 9)))",
        "(sum (list (abs -4) (round 3.14)))",
        "(sum (list (abs -4) (fn [x] (round 3.14))))",
    ]
    for s in examples:
        print(f"S式: {s}")
        print(f"  括弧深さ: {countparen(s)}")
        print(f"  角括弧深さ: {countkakko(s)}")
        print(f"  Dycks  : {sexp_str_to_dyck_and_labels(s)}")
        print()