import argparse
import re
from difflib import context_diff

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
            context_diff(org,exp)
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
    args = ap.parse_args()
    if(args.regression):
        testlist_regression(args.n,args.max_depth,args.max_bind)
    else:
        testlist()