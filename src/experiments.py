import pipeline_cv_train as p

def dprint(s,fp):
    print(s)
    if(type(fp)==list):
        for f in fp:
            print(s,file=f)
    else:   
        print(s,file=fp)


def depth_test(args):
    for depth in range(1,5):
        args.max_depth = depth  # 各S式の最大深さ
        for l in  [1,2,3]:
            args.num_layer = l
#                for use_amp in [False,True]:
            print("params",args)
            p.pipeline_arg(args)
            print("success",args)

def attention_combination(args):
    # S-exp params
    args.n_sexps = 5000  # 生成するS式サンプル数
    args.n_free_vars = 1  # 各S式の自由変数の数

    # Transformer params
    args.d_model = 256  # depth of model
    args.nhead = 4  # num. of heads 
    args.max_depth = 4  # 各S式の最大深さ
    args.force_train=True
    args.use_amp=False
    args.show_msg=False
    args.attentiononly=True
    with open("attentiononly_examples.log","wa") as logfp:
        for noembedded in [False,True]:
            args.noembedded=noembedded
            for recursive in  [True ,False]:
                args.recursive=recursive
                for l in  [1,2,3]:
                    args.num_layer = l
    #                for use_amp in [False,True]:
                    dprint("params",args,logfp)
                    p.pipeline_arg(args)
                    dprint("success",args,logfp)

if __name__ == "__main__":
    args = p.parse_args()
    attention_combination(args)