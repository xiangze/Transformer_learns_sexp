import pipeline_cv_train as p

def attention_combination(args):
    # S-exp params
    args.n_sexps = 5000  # 生成するS式サンプル数
    args.n_free_vars = 1  # 各S式の自由変数の数
    args.max_depth = 2  # 各S式の最大深さ
    # Transformer params
    args.d_model = 256  # depth of model
    args.nhead = 4  # num. of heads 
    args.force_train=True
    args.use_amp=False
    args.show_msg=False
    args.attentiononly=True
    for noembedded in [False,True]:
        args.noembedded=noembedded
        for recursive in  [True ,False]:
            args.recursive=recursive
            for l in  [1,2,3]:
                args.num_layer = l
#                for use_amp in [False,True]:
                print("params",args)
                p.pipeline_arg(args)
                print("success",args)

if __name__ == "__main__":
    args = p.parse_args()
    attention_combination(args)