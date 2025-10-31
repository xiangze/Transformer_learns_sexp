import argparse

#()->[]
listhreads=["("+" "*i+"list" for i in range(10)]
#括弧の数だけを数える
def _replace_list(inss:str,debug=False,addlist=True):
    depth=0
    kakkostack=[]
    listfound=False
    tmp=""
    ins=inss
    while(ins!=""):
         print(len(ins))
         if(debug):
             print(f"depth:{depth}  input:{ins}  outtemp:{tmp}")
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
            elif(s==")"):
                if(depth==kakkostack[-1]):
                    kakkostack.pop()
                    if(addlist):
                        tmp+="])"
                    else:
                        tmp+="]"
                depth-=1
            else:
                tmp+=s
            ins=ins[1:]
    return tmp
def _replace_list(inss:str,debug=False,addlist=True):
    pass

replace_list=lambda s,debug=False,n=10,addlist=True:_replace_list(s,debug=debug,addlist=addlist)[0]

def __replace_list(ins:str,tmp:str="",listdepth:int=0,depths:list=[],lens:list=[],debug=False,addlist=True):
    """
    ins: 入力str
    tmp: 出力str
    i 処理対象文字index
    listdepth []階層の深さ
    depths 各[]階層での()階層の深さ
    lens[] 各[]階層での文字列の長さ
    """
    i=0
    while(ins!=""):
         assert(listdepth>=0)
         assert(depths[listdepth]>=0)
         listnotfound=True
         for ls in listhreads: #"( list"
            l=len(ls)
            if(len(ins)>l and ins[i:i+l]==ls):    
                for ld in range(listdepth+1):lens[ld]+=l
                depths[listdepth]=+1
                depths[listdepth+1]=1
                if(debug):               
                    print(f"listin outtmp {tmp}, lens {lens}")
                tmp,lens =__replace_list(ins[i+l:],"".join([tmp]+[ls]*addlist+[" ["]),listdepth+1,depths,lens,debug,addlist)

                if(debug):               
                    print(f"listout {listdepth+1} list length {lens[listdepth+1]} ")
                i+=l+lens[listdepth+1]
                lens[listdepth+1]=0
                listnotfound=False
                break
            
         if(listnotfound):
            try:
                s=ins[i]
            except:
                print("*",i,len(ins),ins,listdepth,depths[listdepth])
                exit()

            if(s=="("):
                depths[listdepth]+=1
            elif(s==")"):
                depths[listdepth]-=1
                if(depths[listdepth]==0):
                        if(listdepth==0):
                            return tmp+")",lens
                        if(addlist):
                            tmp+="])"
                        else:
                            tmp+="]"

                        if(debug):               
                            print(f"i{i+1}: s={s} listdepth:{listdepth-1} depth:{depths} lens:{lens}, input:{ins[i+1:]}  outtemp:{tmp}")
                            print("close )")
                        i+=1
                        for ld in range(listdepth+1):lens[ld]+=1 
                        return tmp,lens
            tmp+=s
            i+=1
            for ld in range(listdepth+1):lens[ld]+=1 

         if(debug):
             print(f"i{i}: s={s} listdepth:{listdepth} depth:{depths} lens:{lens}, input:{ins[i:]}  outtemp:{tmp}")

    return tmp,lens

#replace_list=lambda s,debug=False,n=10,addlist=True:_replace_list(s,"",0,[0]*n,[0]*n,debug=debug,addlist=addlist)[0]


def test_inout(org,exp,debug=False):
    s=replace_list(org,n=5,debug=debug).replace("[ ","[")
    print(f"in {len(org)}:  {org}")
    print("out:",s)
    print("exp:",exp)
    return s

def testlist(pat=[],debug=False):
    patterns=[
        ["(do (list -6.72 (list (* -4 9) 4 x) ((fn [y] -9) -8.38) v))",
         "(do (list [ -6.72 (list [ (* -4 9) 4 x]) ((fn [y] -9) -8.38) v)] )"],
         ["(do (list a (list x y z) e f) d )",
          "(do (list [a (list [x y z]) e f]) d )"         ],
          ["(do (list a (list x y) e f) d)",
           "(do (list [a (list [x y]) e f]) d)"           ],
           ["(do (list a (list x (list y z)) e f) d )",
            "(do (list [a (list [x (list [y z]) ]) e f]) d )"             ]
        ]
    patterns=[
        ["(do (setv y (sum (list  (min (list  ((fn [v] 3) -4.77) (if (> 8.79 x) -7 t) (pow y 4) x)) (* ((fn [a] 3) 4) (* 4.56 -3)) ((fn [v] (+ -8 v)) (+ 6.8 0.97)) ((fn [c] (pow t -1)) (if (= -3 a) -1 1.06))))))",
         "(do (setv y (sum (list[ (min (list[  ((fn [v] 3) -4.77) (if (> 8.79 x) -7 t) (pow y 4) x)] ]) (* ((fn [a] 3) 4) (* 4.56 -3)) ((fn [v] (+ -8 v)) (+ 6.8 0.97))  ((fn [c] (pow t -1)) (if (= -3 a) -1 1.06)))  ))"],
    ]
    if(pat!=[]):
        patterns=pat
        
    for p in patterns:
        org,exp=p
        s=test_inout(org,exp)
        #assert(exp.replace(" ","")==s.replace(" ",""))
        if(exp.replace(" ","")!=s.replace(" ","")):
            s=test_inout(org,exp,debug=True)    
            exit()            

def testlist_regression(n,max_bind, max_depth,debug=False):        
    #s= replace_sexp(sexp).replace("\"","").replace("hy.models.","")
    #s= re.sub(r"\(fn \(([\w+])\)" ,r"(fn [\1]",dumps(sexp))
    import generate_sexp_with_variable as g
    for i in range(n):
        s=g.replace_sexp(g.gen_program_with_setv(max_bind, max_depth))
        testlist([s],debug)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--max_bind", type=int, default=2)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--testlist", action="store_true")
    ap.add_argument("--onlygen", action="store_true")
    args = ap.parse_args()

#    testlist_regression(args.n,args.max_depth,args.max_bind)
    testlist()