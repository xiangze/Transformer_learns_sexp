import pipeline_cv_train as p
import itertools
import datetime
import sys

def dprint(s,fp):
    print(s)
    if(type(fp)==list):
        for f in fp:
            print(s,file=f)
    elif(fp==None):
        print(s)
    else:   
        print(s,file=fp)

def exec(args,logfp):
    dprint(f"params{args}",logfp)
    p.pipeline_arg(args)    
    try:
        p.pipeline_arg(args)
        dprint(f"success: {args}",logfp)
    except Exception  as e:
        dprint(f"fail: {e}",logfp)
        exit()
    
def depth_test(args):
    args.use_amp=False
    for depth in range(1,5):
        args.max_depth = depth  # 各S式の最大深さ
        for l in  [1,2,3]:
            args.num_layer = l
            exec(args,sys.stdout)

def init_attentiononly_recursive(args,show_msg=True):
    args.n_sexps = 10000  # 生成するS式サンプル数
    args.use_amp=False
    args.show_msg=show_msg
    args.attentiononly=True
    args.noembedded=False
    args.recursive=True
    args.n_eval=3
    # Transformer params
    args.d_model = 256  # depth of model
    args.nhead = 4  # num. of heads 
    args.max_depth = 4  # 各S式の最大深さ
    args.n_free_vars = 1  # 各S式の自由変数の数
    return args

def recursive_embedded(args):
    args=init_attentiononly_recursive(args)
    date= datetime.datetime.now()
    with open("recursive_embedded.log","a") as logfp:
        dprint(f"run {date}",logfp)        
        for kind,d_model,depth,n_free_vars,head,layer in itertools.product(
            ["simple","add","ring","meta"],[128,256,512],
            [2,3,4], [3,4,5], [2,4,8],[2,3,4],):
            # Transformer params
            args.want_kind=kind
            args.max_depth = depth  # 各S式の最大深さ
            args.n_free_vars = n_free_vars  # 各S式の自由変数の数
            args.num_layer = layer
            args.nhead = head  # num. of heads 
            args.d_model = d_model  # depth of model
            exec(args,logfp)

def nonrecursive(args):
    args=init_attentiononly_recursive(args)
    date= datetime.datetime.now()
    args.recursive=False
    with open("nonrecursive.log","a") as logfp:
        dprint(f"run {date}",logfp)        
        for kind, d_model, depth, n_free_vars, head, layer in itertools.product(
            ["simple","add","ring","meta"],[128,256,512],
            [2,3,4], [3,4], [8,16],[2,3,4],):
            # Transformer params
            args.want_kind=kind
            args.max_depth = depth  # 各S式の最大深さ
            args.n_free_vars = n_free_vars  # 各S式の自由変数の数
            args.num_layer = layer
            args.nhead = head  # num. of heads 
            args.d_model = d_model  # depth of model
            args.dim_ff=d_model
            exec(args,logfp)

def attention_combination(args):
    args=init_attentiononly_recursive(args)
    with open("attentiononly_examples.log","a") as logfp:
            for recursive in  [True ,False]:
                args.recursive=recursive
                for l in  [1,2,3,4]:
                    args.num_layer = l
                    exec(args,logfp)

def layers(args,act=True,show_msg=True):
    args=init_attentiononly_recursive(args)
    d_model=256
    head=8
    args.activate=act
    args.nhead = head
    args.d_model = d_model
    args.dim_ff=d_model
    for kind in ["simple","add","ring","meta"]:
        args.want_kind=kind
        for l in range(1,4):
            args.num_layer = l
            exec(args,None)    

def combination(args):
    args=init_attentiononly_recursive(args)
    with open("compbination_examples.log","a") as logfp:
        for attentiononly in  [True ,False]: 
            args.attentiononly=attentiononly
            for recursive in  [True ,False]:
                args.recursive=recursive
                for l in  [1,2,3,4]:
                    args.num_layer = l
                    exec(args,logfp)


if __name__ == "__main__":
    args = p.parse_args()
    #nonrecursive(args)
    #recursive_embedded(args)
    layers(args,True)
