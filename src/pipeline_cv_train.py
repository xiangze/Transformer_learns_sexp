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
import sexp2dick as s2d
import mysexp2dick as mys2d
import matrix_visualizer as vis
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch import tensor,save
import util 
import randhof_with_weight as hof
import numpy as np
import attentiononly as atn
import itertools

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
def make_model(params,model_kind,vocab_size,debug):
    params["pad_id"]=0
    #
    params["vocab_size"]=vocab_size
    if params["attentiononly"]:
        if params["recursive"]:
            model=atn.AttentionOnlyFRecursiveRegressor(params,debug=debug,weightvisible=True)
        else:    
            model=atn.AttentionOnlyRegressor(params,debug=debug)
    elif model_kind == "fixed":
        model=fixed.TransformerRegressor(params,debug=debug)
    elif model_kind == "outQK":
        model=fixed.TransformerRegressor(params,debug=debug,outQK=True)
    else:
        raise ValueError("model_kind must be 'fixed' or 'recursive'.")
    return model


def train_one_fold(model,
                   ds_train, ds_val,
                   epochs: int, batch_size: int,
                   device:str="cuda", use_amp=True,evalperi=100,debug=False
                   ) -> Optional[Any]:
    print(f"Device: {device} amp:{use_amp}")
    pin = (device == "cuda")
    ds_train=[tensor(t) for t in ds_train]
    ds_val  =[tensor(t) for t in ds_val] 
    for d in ds_val:
        print(d.shape,",",end="")
    assert(ds_train[0].shape[1]==ds_train[1].shape[1])
    assert(ds_val[0].shape[1]==ds_val[1].shape[1])
    
    num_workers=1
    train_loader = DataLoader(TensorDataset( *ds_train), batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset( *ds_val), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    evalperi=max(1,epochs//10)
    criterion=nn.MSELoss() #soft
    opt=optim.Adam(model.parameters(), lr=0.05)
    scheduler=optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    train_loss,best_val_loss,last_val_loss=util.traineval(epochs,device,model,train_loader,val_loader,criterion,opt,scheduler,use_amp=use_amp,eval=True,peri=evalperi,debug=debug)
    return model,train_loss,best_val_loss,last_val_loss

simplekinds=["simple","meta","arith","add","meta","ring"]
def genSexps(args):
    t0 = time.time()
    if(args.sexpfilename!=""):
        with open(args.sexpfilename,"r",encoding="utf-8") as f:
            ls=[ line.strip() for line in f.readlines()]
        ls=[k.strip().split(",") for k in ls]
        S, ss, steps = map(list, zip(*ls))
        print(f"[2/5] loaded: {len(S)} samples in {time.time()-t0:.2f}s")
        if(args.max_data_num>0):
            S=S[:min(args.max_data_num,len(S))]
    else:
        print("[2/5] Evaluating Higher Order S-expressions...")
        if(args.want_kind in simplekinds):
            SS=hof.gen_and_eval_simple(args.n_sexps,args.max_depth,seed=args.seed,want_kind=args.want_kind,n_free_vars=args.n_free_vars,debug=args.debug)
        else:
            SS=hof.gen_and_eval(args.n_sexps,args.max_depth,seed=args.seed,want_kind=args.want_kind,n_free_vars=args.n_free_vars)
        with open(f"sexp/sexppair_n{args.n_sexps}_d{args.max_depth}_freevar{args.n_free_vars}_kind{args.want_kind}.txt", "w") as f:
            for s in SS:
                print(f"{s[0]},{s[1]},{s[2]}",file=f)            
        S, ss, steps = map(list, zip(*SS))

    if  steps is not None:
        step_log_path = str(out_root) + "/eval_steps.csv"
        with open(step_log_path,"w") as f:
            f.write("index,steps\n")
            for i,s in enumerate(S):
               f.write(f"org exp len {len(s)},reduced len {len(ss[i])}, {steps[i]}steps\n")
        print(f"step counts saved to: {step_log_path}")
        print(f"max len S",max([len(i) for i in S ]))
        print(f"max len ss",max([len(i) for i in ss ]))
        return S,ss,steps
    else:
        return S,ss,None

def toint(S):
    return [[int(i) for i in s] for s in S]

def loadconvertedfile(convfilename):
    print("try loading ",convfilename)
    S=[];ss=[];masks=[];target_masks=[]
    with open(convfilename) as fp:
        for l in fp.readlines():
            n=l.split("], [")
            for i,s in enumerate([S,ss,masks,target_masks]):
                s.append(n[i].replace("]","").replace("[","").replace("(","").split(", "))
        S,ss,masks,target_masks=[[toint(s) for s in n] for n in [S,ss,masks,target_masks]]
        print(f"read {len(S)} pairs from {convfilename}")
    return S,ss,masks,target_masks

"""
S式集合とその簡約化の集合を[one-hot token,文字列長]size list[list](行列相当)のpairに変換し、vocabrary size()とともに返す
"""
def convert(S,ss,args):
    t0 = time.time()
    if(args.sexpfilename!=""):
        convfilename=f"{args.sexpfilename}_conv.csv"
    else:
        convfilename=f"sexp/sexppair_n{args.n_sexps}_d{args.max_depth}_freevar{args.n_free_vars}_kind{args.want_kind}.txt_conv.csv"
    #if(os.path.isfile(convfilename)):
    try:
        S,ss,masks,target_masks=loadconvertedfile(convfilename)
    except Exception as e :
        print(e)#invalid literal for int() with base 10: ')'
        #maskss=[masks_for_S,masks_for_SS]
        tokenss,_,maskss=mys2d.sexpss_to_tokens(S,ss,show=False)
        S,ss=tokenss
        masks=maskss[0]
        target_masks=maskss[1]
        d=[S,ss,masks,target_masks]
        with open(convfilename, "w") as f: 
            for p in zip(*d):
                print(p,file=f)
    pairs=[list(p) for p in zip(S,ss,masks,target_masks)]
    try:
        vocab_size=max([max(s) for s in S]+[max(s) for s in ss])+1
    except Exception as e :
            print(e)
            print(S)
            print(ss)
            exit()
        #print([max(s) for s in S],[max(s) for s in ss])
    print(f"converted: {len(pairs)} pairs in {time.time()-t0:.2f}s")
    print("length of [S,ss,maksk,target_maksk]=",len(S[0]),len(ss[0]),len(masks),len(target_masks))
    print("vacab size",vocab_size)
    print("len pairs",len(pairs))
    pairs=[[np.array(p[i]) for i in range(len(p))]  for p in pairs]
    return pairs,vocab_size

def pipeline1(args,kind="any"):
    args.n_sexps=1
    args.want_kind=kind
    S,ss,steps=genSexps(args)
    pairs,vocab_size=convert(S,ss,args)
    ds = [tensor(np.array(list(t))) for t in zip(*pairs)]
    mask=tensor(np.ones(ds[2].shape)).to(args.device)#mask=1のとき入力が有効になる
    return ds[0].to(args.device),mask,vocab_size

def pipeline(args,
             params_sexp:dict,
             params_tr: dict ={"d_model":256, "nhead":8, "num_layer" : 4, "dim_ff": 1024, "max_len": 4096,"dropout":0.2},
             out_root="result", seed: int =-1):
    if(args.debug):
        args.n_sexps=10
    print("[1/5] Generating S-expressions...")
    S,ss,steps=genSexps(args)
    print("[3/5] Converting to Dyck language...")
    pairs,vocab_size=convert(S,ss,args)
    params_tr["max_len"]=min(args.max_len,max( [len(s[0]) for s in pairs]))

    print("[4/5] K-fold training/evaluation...")

    pname="".join([f"{k}_{v}_" for k,v in params_tr.items()])
    pname=pname+"".join([f"{k}_{v}_" for k,v in params_sexp.items()])+f"_epoch{args.epochs}"
    pname=pname.replace("sexpfilename__","").replace("want_","").replace("num_free_vars","freevars").replace("num_layer","nlayer").replace("d_model","d_")

    folds = kfold_split(len(pairs), args.kfold, args.seed)[:1]
    with open(f"log/{pname}.log","w") as fpw:
        for k, (tr_idx, va_idx) in enumerate(folds):
            os.makedirs(f"{out_root}/fold_{k+1:02d}", exist_ok=True)
            model=make_model(params_tr,args.model,vocab_size,args.debug).to(args.device)
            train_pairs = [pairs[i] for i in tr_idx]
            val_pairs   = [pairs[i] for i in va_idx]

            ds_train = [np.array(list(t)) for t in zip(*train_pairs)]
            ds_val   = [np.array(list(t)) for t in zip(*val_pairs)]
            if(len(ds_train)>0 and len(ds_val)>0):
                modelname=f"model/{pname}_{k}.pth"
                if(os.path.isfile(modelname) and not args.force_train):
                    try:
                        model.load_state_dict(torch.load(modelname)).to(args.device)
                    except:
                        ckpt = torch.load(modelname, map_location=args.device)
                        state_dict = ckpt if isinstance(ckpt, dict) and "tok.weight" in ckpt else ckpt["state_dict"]
                        vocab_size, d_model = state_dict["tok.weight"].shape
                        #print("ckpt vocab_size, d_model =", vocab_size, d_model)
                        model=make_model(params_tr,args.model,vocab_size,args.debug)
                        model.load_state_dict(state_dict, strict=True)
                        model=model.to(args.device)
                    train_loss,best_val_loss,last_val_loss=1,1,1
                else:
                    model,train_loss,best_val_loss,last_val_loss=train_one_fold(model, ds_train, ds_val,
                                                            epochs=args.epochs, batch_size=args.batch_size,
                                                            device=args.device,use_amp=args.use_amp,evalperi=args.evalperi,debug=args.debug)
                    save(model.state_dict(), modelname)
                msg=f"[5/5][fold {k+1}/{args.kfold}] train loss: {train_loss}, best val loss: {best_val_loss}, last val loss: {last_val_loss}"
                print(msg)
                print(msg,file=fpw)
                print(f"[5/5][fold {k+1}/{args.kfold}] visualizing attention")
                print("--- sample inpuit")
                xin,mask,_vocab_size=pipeline1(args,args.want_kind)
                print("--- sample inpuit(end)")
                #assert(_vocab_size==vocab_size)
                print("xin",xin,xin.shape,"mask",mask)
                vis.save_attention_heatmap(model,params_tr,vocab_size,args.device,f"{pname}_{k}",x=xin,mask=mask,out_dir="img/",getAttention=("outQK"!=params_tr["model"]))
                #vis.show_QKV(model.enc, "QKV_"+pname,params_tr["nhead"],out_dir="img/",device="cuda")
                vis.show_QKV(model.enc, "QKV_"+pname+"rand",params_tr["nhead"],out_dir="img/",device="cuda",x=torch.randn(params_tr["d_model"]))

def run_all(args,out_root):
    for n,depth,n_free_vars,head,layer,kind in itertools.product(
        [8000,10000,20000,50000], [2,3,4], [3,4,5], [2,4,8],[2,3,4],
        ["int","kinder","list","closure","withlet","default"],  ):
        args.want_kind=kind
        params_sexp:dict={"num":n,"num_free_vars":n_free_vars,"max_depth":depth,"sexpfilename":args.sexpfilename,"want_kind":kind}
        params_tr: dict ={"d_model":args.d_model, "nhead":head, "num_layer" :layer, "dim_ff": args.dim_ff, "max_len": args.max_len}
        pipeline(args, params_sexp,params_tr,out_root=out_root)

def run_small(args,out_root,kinds=["simple","add","ring","meta"],
              models=["fixed", "recursive","attentiononly","outQK"]):
    for n,       depth,  n_free_vars,head,layer,   d_model,kind ,model in itertools.product(
        [30000], [1,2,3], [3],      [2,4], [1,2,3],[256],#,512,256+512],
        kinds,models):#["kinder","closure","withlet"]
        args.want_kind=kind
        params_sexp:dict={"num":n,"num_free_vars":n_free_vars,"max_depth":depth,"sexpfilename":args.sexpfilename,"want_kind":kind}
        params_tr: dict ={"d_model":d_model, "nhead":head, "num_layer" :layer, "dim_ff": args.dim_ff, "max_len": args.max_len,"model":model}
        pipeline(args, params_sexp,params_tr,out_root=out_root)


# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="S→eval→Dyck→K-foldでTransformer学習・可視化まで一括実行")
    # S-exp params
    parser.add_argument("--n_sexps", type=int, default=30000, help="生成するS式サンプル数")
    parser.add_argument("--n_free_vars", type=int, default=4, help="各S式の自由変数の数")
    parser.add_argument("--max_depth", type=int, default=4, help="各S式の最大深さ")
    parser.add_argument("--sexpfilename", type=str, default="",help="use sexp from file") #S式をファイルから読み込む
    parser.add_argument("--max_data_num", type=int, default=0)
    parser.add_argument("--want_kind", type=str, default="int")
    # leaning params
    parser.add_argument("--kfold", type=int, default=5, help="交差検証のfold数")
    parser.add_argument("--model", type=str, choices=["fixed", "recursive","attentiononly","outQK"], default="fixed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",help="device(cpu/cuda)")
    parser.add_argument("--evalperi", type=int, default=100,help="evaluation period during training")
    # Transformer params
    parser.add_argument("--d_model",   type=int, default=256, help="depth of model")
    parser.add_argument("--nhead",     type=int, default=8,   help="num. of heads")
    parser.add_argument("--num_layer", type=int, default=4,   help="num. of layers")
    parser.add_argument("--dim_ff",    type=int, default=256, help="dim. of FNN")
    parser.add_argument("--max_len",   type=int, default=4096,help="max length of input sequence")
    parser.add_argument("--dropout",   type=float, default=0.2,help="dropout")
    parser.add_argument("--recursive",   action="store_true")   
    parser.add_argument("--attentiononly",   action="store_true")   
    # others
    parser.add_argument("--output_dir", type=str, default="./runs/exp")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--use_amp", action="store_true") 
    # old
    parser.add_argument("--use_s2d", action="store_true")

    args = parser.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    args.use_amp=args.use_amp and (args.device=="cuda")       
    if(args.all):
        run_all(args,out_root)
    elif(args.simple):
        run_small(args,out_root,["simple"])
    elif(args.small):
        run_small(args,out_root)
    else:
        params_sexp:dict={"num":args.n_sexps,"num_free_vars":args.n_free_vars,"max_depth":args.max_depth,"sexpfilename":args.sexpfilename,"want_kind":args.want_kind}
        if(args.sexpfilename!=""):
            params_sexp["sexpfilename"]=args.sexpfilename
        params_tr: dict ={"d_model":args.d_model, "nhead":args.nhead, "num_layer" : args.num_layer, 
                          "dim_ff": args.dim_ff, "max_len": args.max_len,"dropout":args.dropout,
                          "model":args.model,"recursive":args.recursive,"attentiononly":args.attentiononly}
        pipeline(args, params_sexp,params_tr,out_root=out_root)
