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
    
def sexp_str_to_dyck(s:str,worddict=None,show=False) -> List[int]:
     depth=0
     kdepth=0
     Dyks = []
     maxdepth=countparen(s)
     """S式文字列 s に含まれる単語数をカウント"""
     contents = s.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').split()
     if(worddict is None):
        worddict={token:i for i, token in enumerate(list(set(contents))) }
     tokens = s.replace('(', '( ').replace(')', ' )').replace('[', '[ ').replace(']', ' ]').split()

     if(show):
        print(contents)
        print(worddict)
        print(tokens)
        
     for t in tokens:
        if(t=='('):
            depth+=1
            Dyks.append(depth*2)
        elif(t==')'):
            Dyks.append(depth*2+1)
            depth-=1
        elif(t=='['):
            kdepth+=1
            Dyks.append(-kdepth*2)
        elif(t==']'):
            Dyks.append(-kdepth*2-1)
            kdepth-=1
        else:
            try:
                Dyks.append(worddict[t]+maxdepth)
            except KeyError:
                print(tokens)
                exit()

     mim=min(Dyks)
     Dyks = [d - mim + 1 for d in Dyks]
     Dyks.append(0) # End token
     return Dyks

def one_hot_encode_dyck(Dycks:List[List[int]], vocab_size:int) -> torch.Tensor:
    tensor_list = []
    for dyck in Dycks:
        dyck_tensor = torch.tensor(dyck, dtype=torch.long)
        one_hot_tensor = torch.nn.functional.one_hot(dyck_tensor, num_classes=vocab_size)
        tensor_list.append(one_hot_tensor)
    return torch.stack(tensor_list)  # バッチ次元でスタック
     
def sexps_to_tokens(S:list,padding=False,show=False) -> List[List]:
    contents = []
    for s in S:
        contents.extend(s.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').split())
    worddict={token:i for i, token in enumerate(list(set(contents))) }
    if(padding):
        tokens=[sexp_str_to_dyck(s, worddict=worddict, show=show) for s in S]
        maxlen=max([len(s) for s in tokens])
        tokens=[s+[0]*(maxlen-len(s)) for s in tokens]
        return tokens,worddict
    else:
        return [sexp_str_to_dyck(s, worddict=worddict, show=show) for s in S],worddict

def sexps_to_tokens_onehot(S:list,show=False) -> List[List]:        
    Dycks,_=sexps_to_tokens(S,show=show)
    maxdepth=max([max(dyck) for dyck in Dycks])
    return one_hot_encode_dyck(Dycks,vocab_size=maxdepth+1)

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
        print(f"  Dycks  : {sexp_str_to_dyck(s)}")
        print("one hot",one_hot_encode_dyck([sexp_str_to_dyck(s)],vocab_size=20))