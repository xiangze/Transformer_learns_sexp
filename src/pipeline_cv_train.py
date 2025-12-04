"""
- 処理:
  S式の生成(S) → 評価(ss) → Dyck変換(D, dd) → K-fold交差検証で学習・評価
  モデルは transformer_dick_fixed_embed.py（--model fixed）
           または Recursive_Transformere.py（--model recursive）
  学習後に matrix_visualizer.py があればAttention行列を保存（--visualize）
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import util
import generate_sexp_with_variable as gen_mod
import evallisp as eval_mod
import sexp2dick as s2d
import mysexp2dick as mys2d
import matrix_visualizer as vis
import step_counter
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch import tensor
import util 
import randhof_with_weight as hof
import numpy as np
#import attentiononly as atn
# ------------------------------
# Data pipeline components
# ------------------------------
def eval_S(S: List[List[str]],use_gen:bool ,log_steps: bool) -> Tuple[List[List[str]], Optional[List[int]]]:
    if(use_gen):
        return gen_mod.hy_eval_program_str(S)
    if(log_steps):
        return step_counter.eval_with_steps(S) 
    else:
        return eval_mod.execSexp(S),None

# ------------------------------
# Dataset helpers
# ------------------------------
def to_str_seq(tokens: List[str]) -> str:
    return " ".join(map(str, tokens))

def make_pairs(D: List[List[str]], dd: List[List[str]]) -> List[Tuple[str, str]]:
    assert len(D) == len(dd)
    return [(to_str_seq(x), to_str_seq(y)) for x, y in zip(D, dd)]


def kfold_split(n: int, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = []
    fold_size = max(1, n // k)
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else (i + 1) * fold_size
        val_idx = idx[start:end]
        train_idx = idx[:start] + idx[end:]
        folds.append((train_idx, val_idx))
    return folds

def save_pairs_jsonl(pairs: List[Tuple[str, str]], path: Path) -> None:
    #os.makedirs(path, exist_ok=True)
    with open(path,"w", encoding="utf-8") as f:
        for x, y in pairs:
            f.write(json.dumps({"input": x, "label": y}, ensure_ascii=False) + "\n")

# ------------------------------
# Training & visualization
# ------------------------------
import transformer_dick_fixed_embed as fixed
import Recursive_Ttansformer as recursive
def train_one_fold(model_kind: str,
                   ds_train, ds_val,
                   epochs: int, batch_size: int, vocab_size: int,
                   params: dict ={"d_model":256, "nhead":8, "num_layer" : 4, "dim_ff": 1024, "max_len": 4096},
                   device:str="cuda", use_amp=True,
                   ) -> Optional[Any]:
    
    print(f"Device: {device} amp:{use_amp}")
    pin = (device == "cuda")
    if model_kind == "fixed":
        model=fixed.TransformerRegressor(vocab_size=vocab_size, d_model=params["d_model"], nhead=params["nhead"], num_layers = params["num_layer"], dim_ff= params["dim_ff"], max_len= params["max_len"])
    elif model_kind == "recursive":
        model=recursive.SharedTransformerRegressor(vocab_size=vocab_size, d_model=params["d_model"], nhead=params["nhead"], num_layers = params["num_layer"], dim_ff= params["dim_ff"], max_len= params["max_len"])
    # elif model_kind == "attentiononly":
    #     model=atn.AttentionOnlyRegressor(d_model=params["d_model"], n_heads=params["nhead"], num_layers=params["num_layer"], dropout=0.1)
    else:
        raise ValueError("model_kind must be 'fixed' or 'recursive'.")

    ds_train=[tensor(t) for t in ds_train]
    ds_val  =[tensor(t) for t in ds_val]
    if(ds_train[0].shape[1]!=ds_train[1].shape[1]):
         print("ds_train",ds_train[0].shape,ds_train[1].shape,ds_train[2].shape)
         exit()

    ds_train = TensorDataset( *ds_train)
    ds_val   = TensorDataset( *ds_val)

    num_workers=1
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    criterion=nn.MSELoss() #soft
    opt=optim.Adam(model.parameters(), lr=0.05)
    scheduler=optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    best_val_acc,last_val_acc=util.traineval(epochs,device,model,train_loader,val_loader,criterion,opt,scheduler,use_amp=use_amp,eval=True)
    return best_val_acc,last_val_acc

def genSexps(args):
    t0 = time.time()
    if(args.sexpfilename!=""):
        with open(args.sexpfilename,"r",encoding="utf-8") as f:
            ls=[ line.strip() for line in f.readlines()]
        ls=[k.strip().split(",") for k in ls]
        S=[r[0] for r in  ls]
        ss=[r[1] for r in ls]
        steps=[r[2] for r in  ls]
        print(f"  loaded: {len(S)} samples in {time.time()-t0:.2f}s")
        if(args.max_data_num>0):
            S=S[:min(args.max_data_num,len(S))]
    elif(args.use_gensexp):
        S=[ gen_mod.gen_program_with_setv_s(max_bind=args.n_free_vars, max_depth=args.max_depth) for s in range(args.n_sexps)]
        print(f"  generated: {len(S)} samples in {time.time()-t0:.2f}s")
        print(S[0])
        print("[2/5] Evaluating S-expressions...")
        t0 = time.time()
        ss, steps = eval_S(S, args.use_gen,args.log_eval_steps)
        print(f"  evaluated: {len(ss)} samples in {time.time()-t0:.2f}s")
    else:
        print("[2/5] Evaluating Higher Order S-expressions...")
        S=hof.gen_and_eval(args.n_sexps,args.max_depth,seed=args.seed)
        with open(f"sexppair_n{args.n_sexps}_d{args.max_depth}_freevar{args.n_free_vars}_kindint.txt", "w") as f:
            for s in S:
                print(f"{s[0]},{s[1]},{s[2]}",file=f)            

        ss=[r[1] for r in S]
        steps=[r[2] for r in  S]
        S=[r[0] for r in  S]

    if  steps is not None:
        step_log_path = str(out_root) + "/eval_steps.csv"
        with open(step_log_path,"w", encoding="utf-8") as f:
            f.write("index,steps\n")
            for i,s in enumerate(S):
               f.write(f"org exp len {len(s)},reduced len {len(ss[i])}, {steps[i]}steps\n")
        print(f"step counts saved to: {step_log_path}")
        print(f"max len S",max([len(i) for i in S ]))
        print(f"max len ss",max([len(i) for i in ss ]))
        return S,ss,steps
    else:
        return S,ss,None

def convert(S,ss,args):
    t0 = time.time()
    convfilename=f"{args.sexpfilename}_conv.csv"
    if(os.path.isfile(convfilename)):
        print("load ",convfilename)
        S=[];ss=[];paddings=[]
        with open(convfilename) as fp:
            ls=fp.readlines()
            for l in ls:
                n=l.split("], [")
                S.append(n[0].replace("[","").split(", "))
                ss.append(n[1].split(", "))
                paddings.append(n[2].replace("]","").split(", "))
        S=[[int(i) for i in s] for s in S]
        ss=[[int(i) for i in s] for s in ss]
        paddings=[[int(i) for i in s] for s in paddings]
        vocab_size=max([max(s) for s in S]+[max(s) for s in ss])
        print("length S,ss,padding",len(S[0]),len(ss[0]),len(paddings[0]))
        print("vacab size",vocab_size)
        pairs=[list(p) for p in zip(S,ss,paddings)]
    elif(not args.use_s2d):
         Dyks,worddict,paddings  = mys2d.sexps_to_tokens(S,padding=True)
         ssDyks,wdss,sspaddings= mys2d.sexps_to_tokens(ss,padding=True)
         worddict.update(wdss)
         vocab_size=len(worddict)+1
         pairs=[list(p) for p in zip(Dyks,ssDyks,paddings)]
         with open(f"sexppair_n{args.n_sexps}_d{args.max_depth}_freevar{args.n_free_vars}_kindint.txt_conv.csv", "w") as f:
             for p in pairs:
                print(p,file=f)
    else:
         Dyks  = s2d.sexp_str_to_dyck_and_labels(S) 
         ssDyks= s2d.sexp_str_to_dyck_and_labels(ss) 
         vocab_size=1000
         pairs = make_pairs(Dyks, ssDyks,paddings)
    pairs=[[np.array(p[i]) for i in range(len(p))]  for p in pairs]
    print(f"  converted: {len(pairs)} pairs in {time.time()-t0:.2f}s")
    return pairs,vocab_size

def pipeline(args,
             params_sexp:dict,
             params_tr: dict ={"d_model":256, "nhead":8, "num_layer" : 4, "dim_ff": 1024, "max_len": 4096},
             out_root="result", seed: int =-1):
    if(args.debug):
        args.n_sexps=10
    print("[1/5] Generating S-expressions...")
    S,ss,steps=genSexps(args)
    print("[3/5] Converting to Dyck language...")
    pairs,vocab_size=convert(S,ss,args)
    params_tr["max_len"]=min(args.max_len,max( [len(s[0]) for s in pairs]))

    print("max S length",params_tr["max_len"])
    print("max ss len",max( [len(s[1]) for s in pairs]))
    print("max vocab size",vocab_size)

    print("[4/5] K-fold training/evaluation...")
    folds = kfold_split(len(pairs), args.kfold, args.seed)
    for k, (tr_idx, va_idx) in enumerate(folds):
        fold_dir = f"{out_root}/fold_{k+1:02d}"
        os.makedirs(fold_dir, exist_ok=True)
        train_pairs = [pairs[i] for i in tr_idx]
        val_pairs   = [pairs[i] for i in va_idx]

        print(f"  [fold {k+1}/{args.kfold}] train={len(train_pairs)} val={len(val_pairs)}")
        if(not args.use_s2d):
            ## "transpose"
            ds_train = [np.array(list(t)) for t in zip(*train_pairs)]
            ds_val   = [np.array(list(t)) for t in zip(*val_pairs)]
        else:
            ds_train = fixed.ExprDataset(train_pairs, mode="dyck")
            ds_val   = fixed.ExprDataset(val_pairs,   mode="dyck")

        print("[4/5] token to Tensor...")
        best_val_acc,last_val_acc=train_one_fold(args.model, ds_train, ds_val,
                                                 epochs=args.epochs, batch_size=args.batch_size, vocab_size=vocab_size,params=params_tr,
                                                 device=args.device,use_amp=(args.device=="cuda"))

        print(f"[fold {k+1}] best val acc: {best_val_acc}, last val acc: {last_val_acc}")
        if args.visualize:
            print(f"  [fold {k+1}] visualizing attention (if supported)...")
            vis.print_and_dump_attention_params(args.model,)
        print("[5/5] Done.")

# ------------------------------
# Main
# ------------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="S→eval→Dyck→K-foldでTransformer学習・可視化まで一括実行")
    # S-exp params
    parser.add_argument("--n_sexps", type=int, default=5000, help="生成するS式サンプル数")
    parser.add_argument("--n_free_vars", type=int, default=2, help="各S式の自由変数の数")
    parser.add_argument("--max_depth", type=int, default=10, help="各S式の最大深さ")
    parser.add_argument("--sexpfilename", type=str, default="",help="use sexp from file") #S式をファイルから読み込む
    parser.add_argument("--max_data_num", type=int, default=0)
    # leaning params
    parser.add_argument("--kfold", type=int, default=5, help="交差検証のfold数")
    parser.add_argument("--model", type=str, choices=["fixed", "recursive"], default="fixed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",help="device(cpu/cuda)")
    # Transformer params
    parser.add_argument("--d_model",   type=int, default=256, help="depth of model")
    parser.add_argument("--nhead",     type=int, default=8,   help="num. of heads")
    parser.add_argument("--num_layer", type=int, default=4,   help="num. of layers")
    parser.add_argument("--dim_ff",    type=int, default=256, help="dim. of FNN")
    parser.add_argument("--max_len",   type=int, default=4096,help="max length of input sequence")
    # others
    parser.add_argument("--output_dir", type=str, default="./runs/exp")
    parser.add_argument("--log_eval_steps", action="store_true",  help="evallistがステップ数を返すAPIを持つ場合にCSV保存")
    parser.add_argument("--visualize", action="store_true",help="学習後にmatrix_visualizerでAttention Matrixを保存 (可能な場合)")
    parser.add_argument("--debug", action="store_true")
    # old
    parser.add_argument("--use_s2d", action="store_true")
    parser.add_argument("--use_gensexp", action="store_true",help="use old sexp generator")
    
    args = parser.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    params_sexp:dict={"num":args.n_sexps,"num_free_vars":args.n_free_vars,"max_depth":args.max_depth,"sexpfilename":args.sexpfilename}
    params_tr: dict ={"d_model":args.d_model, "nhead":args.nhead, "num_layer" : args.num_layer, "dim_ff": args.dim_ff, "max_len": args.max_len}
    pipeline(args, params_sexp,params_tr,out_root=out_root)
