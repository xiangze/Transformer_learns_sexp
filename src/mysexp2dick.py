from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch

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
    
def sexp_str_to_tokens(s:str,worddict=None,show=False) -> List[int]:
     depth=0
     kdepth=0
     maxdepth=countparen(s)
     """S式文字列 s に含まれる単語数をカウント"""
     
     if(worddict is None):
        worddict=makedict([s])

     tokens = s.replace('(', '( ').replace(')', ' )').replace('[', '[ ').replace(']', ' ]').split()

     if(show):
        print(worddict)
        print(tokens)
     ts=[]
     for t in tokens:
        if(t=='('):
            depth+=1
            ts.append(depth*2)
        elif(t==')'):
            ts.append(depth*2+1)
            depth-=1
        elif(t=='['):
            kdepth+=1
            ts.append(-kdepth*2)
        elif(t==']'):
            ts.append(-kdepth*2-1)
            kdepth-=1
        else:
            try:
                ts.append(worddict[str(t)]+maxdepth)
            except KeyError:
                print(f"no key {t}, tokens{tokens}, ts {ts}, worddict {worddict}")
                exit()
     ts=[d - min(ts) + 1 for d in ts]
     return ts #.append(0) # End token0

def sexp_str_to_dyck(s:str,worddict=None,show=False) -> List[int]:
    depth=0
    kdepth=0
    _stack=[]
    Dycks = s.replace('( ', '(').replace(')', ' )').replace('[ ', '[').replace(']', ' ]').split()
    if(show):
        print(worddict)
        print(Dycks)
    ts=[]
    for t in Dycks:
        if('(' in t):
            depth+=1
            ts.append((worddict[t],depth*2))
            _stack.append(worddict[t])
        elif(t==')'):
            ts.append((_stack.pop(),depth*2+1))
            depth-=1
        elif(t=='['):
            kdepth+=1
            ts.append(worddict[t],-kdepth*2)
            _stack.append(worddict[t])
        elif(t==']'):
            ts.append(_stack.pop(),-kdepth*2-1)
            kdepth-=1
        else:
            try:
                ts.append(worddict[t],0)
            except KeyError:
                print(f"no key {t}",ts)
                exit()
    return ts

def one_hot_encode_dyck(Dycks:List[List[int]], vocab_size:int) -> torch.Tensor:
    tensor_list = []
    for dyck in Dycks:
        dyck_tensor = torch.tensor(dyck, dtype=torch.long)
        one_hot_tensor = torch.nn.functional.one_hot(dyck_tensor, num_classes=vocab_size)
        tensor_list.append(one_hot_tensor)
    return torch.stack(tensor_list)  # バッチ次元でスタック

def makedict(S:list):
    contents = []
    for s in S:
        contents.extend(s.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').split())
    return {token:i for i, token in enumerate(list(set(contents))) }

def padding(tokens,maxlen):
    return [s+[0]*(maxlen-len(s)) for s in tokens]

def sexps_to_tokens(S:list,padding=False,show=False) -> List[List]:
    worddict=makedict(S)
    if(padding):
        tokens=[sexp_str_to_tokens(s, worddict=worddict, show=show) for s in S]
        maxlen=max([len(s) for s in tokens])
        masks=[[int(i<=len(s)) for i in range(maxlen) ] for s in tokens]
        tokens=padding(tokens,maxlen)
        print("maxlen",maxlen)
        return tokens,worddict,masks
    else:
        return [sexp_str_to_tokens(s, worddict=worddict, show=show) for s in S],worddict,None

def sexpss_to_tokens(S1:list,S2:list,show=False) -> List:
    worddict=makedict(S1)
    worddict.update(makedict(S2))
    tokenss=[ [sexp_str_to_tokens(s, worddict=worddict, show=show) for s in k] for k in [S1,S2]]
    maxlen=max([len(sk) for tokens in tokenss for sk in tokens])
    maskss=[[[int(i>=len(s)) for i in range(maxlen)] for s in ss ]for ss in [S1,S2]] 
    #tokenss=[ padding(tokens,maxlen) for tokens in tokenss]
    tokenss=[[s+[0]*(maxlen-len(s)) for s in tokens] for tokens in tokenss]
    return tokenss,worddict, maskss

def sexps_to_tokens_onehot(S:list,show=False) -> List[List]:        
    Dycks,_=sexps_to_tokens(S,show=show)
    maxdepth=max([max(dyck) for dyck in Dycks])
    return one_hot_encode_dyck(Dycks,vocab_size=maxdepth+1)

def example():
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
        print(f"  Dycks  : {sexp_str_to_tokens(s)}")
        print("one hot",one_hot_encode_dyck([sexp_str_to_tokens(s)],vocab_size=20))

if __name__ == "__main__":
    example()
