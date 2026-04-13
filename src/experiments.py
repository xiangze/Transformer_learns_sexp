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

def init(args,force_train=False,show_msg=True):
    args.n_sexps = 10000  # 生成するS式サンプル数
    args.force_train=force_train
    args.use_amp=False
    args.show_msg=show_msg
    args.attentiononly=True
    args.noembedded=False
    args.recursive=True
    args.n_eval=10
    # Transformer params
    args.d_model = 256  # depth of model
    args.nhead = 4  # num. of heads 
    args.max_depth = 4  # 各S式の最大深さ
    args.n_free_vars = 1  # 各S式の自由変数の数
    return args

def recursive_embedded(args,force_train=False,show_msg=False):
    args=init(args)
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

def attention_combination(args):
    # S-exp params
    args=init(args)
    with open("attentiononly_examples.log","a") as logfp:
        for noembedded in [False,True]:
            args.noembedded=noembedded
            for recursive in  [True ,False]:
                args.recursive=recursive
                for l in  [2,3]:
                    args.num_layer = l
                    exec(args,logfp)

def layers(args,show_msg=True):
    args=init(args)
    for l in range(1,5):
        args.num_layer = l
        exec(args,None)    

if __name__ == "__main__":
    args = p.parse_args()
    #attention_combination(args)
    #recursive_embedded(args,force_train=False,show_msg=True)
    layers(args,True)
    # for kind,d_model,depth,n_free_vars,head,layer in itertools.product(
    #     ["simple","add","ring","meta"],[128,256,512],
    #     [2,3,4], [3,4,5], [2,4,8],[2,3,4],):
    #     print(f"kind{kind},d_model{d_model},depth{depth},n_free_vars{n_free_vars},head{head},layer{layer}")
