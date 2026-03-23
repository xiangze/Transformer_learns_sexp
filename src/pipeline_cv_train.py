"""
- 処理:
  S式の生成(S) → 評価(ss) → Dyck変換(D, dd) → K-fold交差検証で学習・評価
  モデルは transformer_dick_fixed_embed.py（--model fixed）
           または Recursive_Transformere.py（--model recursive）
  matrix_visualizer.py でAttention行列を可視化、保存
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from dataclasses import dataclass
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

@dataclass
class PipelineArgs:
    # S-exp params
    n_sexps: int = 30000  # 生成するS式サンプル数
    n_free_vars: int = 4  # 各S式の自由変数の数
    max_depth: int = 4  # 各S式の最大深さ
    sexpfilename: str = ""  # use sexp from file
    max_data_num: int = 0  # 最大データ数（0で無制限）
    want_kind: str = "int"  # 生成したい式の種類
    # learning params
    kfold: int = 5  # 交差検証のfold数
    model: str = "fixed"  # モデル種別: fixed/recursive/attentiononly/outQK
    epochs: int = 100  # 学習エポック数
    batch_size: int = 64  # バッチサイズ
    seed: int = 42  # 乱数シード
    device: str = "cuda"  # device(cpu/cuda)
    evalperi: int = 100  # evaluation period during training
    # Transformer params
    d_model: int = 256  # depth of model
    nhead: int = 8  # num. of heads
    num_layer: int = 4  # num. of layers
    dim_ff: int = 256  # dim. of FNN
    max_len: int = 4096  # max length of input sequence
    dropout: float = 0.2  # dropout
    recursive: bool = False  # recursive attention variant flag
    attentiononly: bool = False  # attention-only model flag
    noembedded: bool = False  # disable token embedding
    # others
    n_eval: int = 2  # eval num
    output_dir: str = "./runs/exp"  # output directory
    debug: bool = False  # debug mode
    all: bool = False  # run all experiment grid
    small: bool = False  # run reduced experiment grid
    force_train: bool = False  # retrain even if checkpoint exists
    simple: bool = False  # run only simple kind grid
    use_amp: bool = False  # mixed precision training
    test_attention:bool=False
    show_msg:bool=True
    # old
    use_s2d: bool = False  # legacy flag


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineArgs()
    parser = argparse.ArgumentParser(description="S→eval→Dyck→K-foldでTransformer学習・可視化まで一括実行")
    # S-exp params
    parser.add_argument("--n_sexps", type=int, default=defaults.n_sexps, help="生成するS式サンプル数")
    parser.add_argument("--n_free_vars", type=int, default=defaults.n_free_vars, help="各S式の自由変数の数")
    parser.add_argument("--max_depth", type=int, default=defaults.max_depth, help="各S式の最大深さ")
    parser.add_argument("--sexpfilename", type=str, default=defaults.sexpfilename, help="use sexp from file")
    parser.add_argument("--max_data_num", type=int, default=defaults.max_data_num)
    parser.add_argument("--want_kind", type=str, default=defaults.want_kind)
    # learning params
    parser.add_argument("--kfold", type=int, default=defaults.kfold, help="交差検証のfold数")
    parser.add_argument(
        "--model",
        type=str,
        choices=["fixed", "recursive", "attentiononly", "outQK"],
        default=defaults.model,
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device, help="device(cpu/cuda)")
    parser.add_argument("--evalperi", type=int, default=defaults.evalperi, help="evaluation period during training")
    # Transformer params
    parser.add_argument("--d_model", type=int, default=defaults.d_model, help="depth of model")
    parser.add_argument("--nhead", type=int, default=defaults.nhead, help="num. of heads")
    parser.add_argument("--num_layer", type=int, default=defaults.num_layer, help="num. of layers")
    parser.add_argument("--dim_ff", type=int, default=defaults.dim_ff, help="dim. of FNN")
    parser.add_argument("--max_len", type=int, default=defaults.max_len, help="max length of input sequence")
    parser.add_argument("--dropout", type=float, default=defaults.dropout, help="dropout")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--attentiononly", action="store_true")
    parser.add_argument("--noembedded", action="store_true")
    # others
    parser.add_argument("--n_eval", type=int, default=defaults.n_eval, help="eval num")
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--test_attention", action="store_true")
    parser.add_argument("--show_msg", action="store_false")
    # old
    parser.add_argument("--use_s2d", action="store_true")
    return parser


def parse_args() -> PipelineArgs:
    parser = build_parser()
    ns = parser.parse_args()
    return PipelineArgs(**vars(ns))

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
    embedding=not params["noembedded"]
    params["pad_id"]=0 ##?
    params["vocab_size"]=vocab_size
    if(debug):
        print("batch_size",params["batch_size"])
        print("d_model",params["d_model"])
        print("seq_len",params["seq_len"])
        print("vocab_size",vocab_size)
    
    
    if params["attentiononly"]:
        if params["recursive"]:
            model=atn.AttentionOnlyRecursiveRegressor(params,debug=debug,weightvisible=True,embedding=embedding)
        else:    
            model=atn.AttentionOnlyRegressor(params,debug=debug,embedding=embedding)
    else:
        if model_kind == "recursive":
            model=fixed.TransformerRegressor(params,debug=debug,recursive=True)
        elif model_kind == "fixed":
            model=fixed.TransformerRegressor(params,debug=debug)
        elif model_kind == "outQK":
            model=fixed.TransformerRegressor(params,debug=debug,outQK=True)
        else:
            raise ValueError("model_kind must be 'fixed' or 'recursive'.")
    return model

def train_one_fold(args,
                   model,
                   ds_train, ds_val,
                   epochs: int, batch_size: int,
                   device:str="cuda", use_amp=True,evalperi=100,debug=False,
                   num_workers=1
                   ) -> Optional[Any]:
    if(args.show_msg):
        print(f"Device: {device} amp:{use_amp}")
    pin = (device == "cuda")
    for d in [ds_train,ds_val]:
        print(d[0].shape)

    for d in [ds_train,ds_val]:
        assert(d[0].shape==d[1].shape) 
        for i in range(len(d)):
            assert(torch.any(d[i][2:4,:]!=0)), print(d[i])

    train_loader = DataLoader(TensorDataset(ds_train), batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(ds_val), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    evalperi=max(1,epochs//10)
    criterion=nn.MSELoss() #soft
    opt=optim.Adam(model.parameters(), lr=0.05)
    scheduler=optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    train_loss,best_val_loss,last_val_loss=util.traineval(epochs,device,model,train_loader,val_loader,criterion,opt,scheduler,use_amp=use_amp,eval=True,peri=evalperi,debug=debug)
    return model,train_loss,best_val_loss,last_val_loss

simplekinds=["simple","meta","arith","add","meta","ring"]
def genSexps(args,out_root,dump=True):
    t0 = time.time()
    if(args.sexpfilename!=""):
        with open(args.sexpfilename,"r",encoding="utf-8") as f:
            ls=[ line.strip() for line in f.readlines()]
        ls=[k.strip().split(",") for k in ls]
        S, ss, steps = map(list, zip(*ls))
        if args.show_msg:
            print(f"[2/5] loaded: {len(S)} samples in {time.time()-t0:.2f}s")
        if(args.max_data_num>0):
            S=S[:min(args.max_data_num,len(S))]
    else:
        if args.show_msg:
            print("[2/5] Evaluating Higher Order S-expressions...")
        if(args.want_kind in simplekinds):
            SS=hof.gen_and_eval_simple(args.n_sexps,args.max_depth,seed=args.seed,want_kind=args.want_kind,n_free_vars=args.n_free_vars,debug=args.debug)
        else:
            SS=hof.gen_and_eval(args.n_sexps,args.max_depth,seed=args.seed,want_kind=args.want_kind,n_free_vars=args.n_free_vars)
        if(dump):
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
        if args.show_msg:
            print(f"step counts saved to: {step_log_path}")
            print(f"max len S",max([len(i) for i in S ]))
            print(f"max len ss",max([len(i) for i in ss ]))
        return S,ss,steps
    else:
        return S,ss,None

def toint(S):
    return [[int(i) for i in s] for s in S]

def loadconvertedfile(args,convfilename):
    if args.show_msg:
        print("try loading ",convfilename)
    S=[];ss=[];masks=[];target_masks=[]
    with open(convfilename) as fp:
        for l in fp.readlines():
            n=l.split("], [")
            for i,s in enumerate([S,ss,masks,target_masks]):
                s.append(n[i].replace("]","").replace("[","").replace("(","").split(", "))
        S,ss,masks,target_masks=[[toint(s) for s in n] for n in [S,ss,masks,target_masks]]
        if args.show_msg:
            print(f"read {len(S)} pairs from {convfilename}")
    return S,ss,masks,target_masks

"""
S式集合とその簡約化の集合を[one-hot token,文字列長]size list[list](行列相当)のpairに変換し、vocabrary size()とともに返す
"""
# ---------------------------------------------------------------------------
# convert キャッシュ読み書き
# ---------------------------------------------------------------------------
def _build_cache_path(args) -> str:
    """コマンドライン引数からキャッシュファイルのパスを決定する。"""
    if args.sexpfilename:
        return f"{args.sexpfilename}_conv.json"
    return (
        f"sexp/sexppair_n{args.n_sexps}_d{args.max_depth}"
        f"_freevar{args.n_free_vars}_kind{args.want_kind}.txt_conv.json"
    )
def save_converted(
    path: str,
    src_tokens: List[List[int]],
    tgt_tokens: List[List[int]],
    src_masks: List[List[int]],
    tgt_masks: List[List[int]],
) -> None:
    """変換済みデータを JSON で書き出す。"""
    data = {
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        "src_masks": src_masks,
        "tgt_masks": tgt_masks,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[cache] saved {len(src_tokens)} pairs → {path}")

def load_converted( path: str, ) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """キャッシュ JSON を読み込み、4 つのリストを返す。
    ファイルが存在しない・壊れている場合は例外を送出する。
    """
    print(f"[cache] loading {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = {"src_tokens", "tgt_tokens", "src_masks", "tgt_masks"}
    if not required_keys.issubset(data.keys()):
        raise ValueError(f"キャッシュに必要なキーがありません: {required_keys - data.keys()}")
    src_tokens = [list(map(int, row)) for row in data["src_tokens"]]
    tgt_tokens = [list(map(int, row)) for row in data["tgt_tokens"]]
    src_masks  = [list(map(int, row)) for row in data["src_masks"]]
    tgt_masks  = [list(map(int, row)) for row in data["tgt_masks"]]
    print(f"[cache] read {len(src_tokens)} pairs from {path}")
    return src_tokens, tgt_tokens, src_masks, tgt_masks

def convert(
    src_sexps: List[str],  tgt_sexps: List[str],  args,
) -> Tuple[List[List[np.ndarray]], int]:
    """S式ペアをモデル入力用のトークン行列ペアに変換する。

    1. キャッシュがあれば読み込む（JSON 形式）。
    2. なければトークン化 → キャッシュ保存。
    3. vocabulary size を算出して返す。

    Args:
        src_sexps:  元のS式文字列リスト
        tgt_sexps:  簡約化後のS式文字列リスト
        args:       コマンドライン引数（キャッシュパス生成用）

    Returns:
        pairs:      各要素が [src_tokens, tgt_tokens, src_mask, tgt_mask] の
                    np.ndarray リスト
        vocab_size: 語彙サイズ（最大トークン ID + 1）
    """
    t0 = time.time()
    cache_path = _build_cache_path(args)
    # --- キャッシュ読み込み or 新規変換 ---
    if os.path.isfile(cache_path):
        try:
            src_tokens, tgt_tokens, src_masks, tgt_masks = load_converted(cache_path)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[cache] 読み込み失敗 ({e})。再変換します。")
            src_tokens, tgt_tokens, src_masks, tgt_masks = _tokenize_and_cache(src_sexps, tgt_sexps, cache_path )
    else:
        src_tokens, tgt_tokens, src_masks, tgt_masks = _tokenize_and_cache(src_sexps, tgt_sexps, cache_path )
        
    if args.show_msg:
        print(f"src_tokens,{np.array(src_tokens).shape}, tgt_tokens {np.array(tgt_tokens).shape},\
            src_masks {np.array(src_masks).shape}, tgt_masks {np.array(tgt_masks).shape}")
    
    # --- vocabulary size ---
    all_token_maxes = [max(seq) for seq in src_tokens] + [max(seq) for seq in tgt_tokens]
    vocab_size = max(all_token_maxes) + 1
    # --- ペア化 & numpy 変換 ---
    assert(np.any(src_masks!=0))
    assert(np.any(tgt_masks!=0))
    pairs = np.array([ [np.array(s), np.array(t), np.array(sm,dtype=float), np.array(tm,dtype=float)]
                for s, t, sm, tm in zip(src_tokens, tgt_tokens, src_masks, tgt_masks)])

    elapsed = time.time() - t0
    seq_len = len(src_tokens[0]) if src_tokens else 0
    if(args.noembedded):
        pairs[0][2]=pairs[0][2].repeat(len(pairs[0][2]))
        pairs[0][3]=pairs[0][3].repeat(len(pairs[0][3]))

    for i in range(len(pairs)):
        assert(np.any(pairs[i,2:4,:]>0)),print(pairs[i])
    #attn_mask バイナリマスクの場合、True は対応する位置がアテンションの対象にならないことを示します。
    #key_padding_mask バイナリマスクの場合、True を指定すると、対応するキー値はアテンション処理において無視されます。
    if args.show_msg:
        print(f"[convert] {pairs.shape} pairs, seq_len={seq_len}, vocab_size={vocab_size}, src_mask={pairs[0][2].shape}, attn_mask={pairs[0][3].shape} {elapsed:.2f}s")
    return pairs, vocab_size,seq_len

def _tokenize_and_cache(src_sexps: List[str], tgt_sexps: List[str],  cache_path: str,) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """トークン化を実行し、結果をキャッシュに保存して返す。"""
    tokenss, _worddict, maskss = mys2d.sexpss_to_tokens(src_sexps, tgt_sexps, show=False)
    src_tokens, tgt_tokens = tokenss
    src_masks, tgt_masks = maskss
    save_converted(cache_path, src_tokens, tgt_tokens, src_masks, tgt_masks)
    return src_tokens, tgt_tokens, src_masks, tgt_masks

def pipeline1(args,out_root,kind="any"):
    n_sexps=args.n_sexps
    args.n_sexps=1
    args.want_kind=kind
    S,ss,steps=genSexps(args,out_root,dump=False)
    pairs,vocab_size,seq_len=convert(S,ss,args)
    ds = [tensor(np.array(list(t))) for t in zip(*pairs)]
    mask=tensor(np.ones(ds[2].shape)).to(args.device)#mask=1のとき入力が有効になる assert(target_mask!=0).any(), f"target_mask, {target_mask}"
    assert(torch.any(mask!=0))
    args.n_sexps=n_sexps
    return ds[0].to(args.device),mask,vocab_size

def eval_show(args,params_tr,model,out_root,i,vocab_size,pname,k):
    if args.show_msg:
        print(f"--- sample input {i}/{args.n_eval}")
    xin,mask,_vocab_size=pipeline1(args,out_root,args.want_kind)
    xout=model(xin,mask)
    assert(not torch.isnan(xout).any()),f"xout{xout}"
    vis.save_attention_heatmap(model,params_tr,vocab_size,args.device,f"{pname}_{k}_{i}",x=xin,mask=mask,out_dir="img/",getAttention=("outQK"!=params_tr["model"]))
    try:
        #vis.show_QKV(model.enc, "QKV_"+pname,params_tr["nhead"],out_dir="img/",device="cuda")
        vis.show_QKV(model.enc, f"QKV_{pname}_{k}_{i}",params_tr["nhead"],out_dir="img/",device="cuda",x=torch.randn(params_tr["d_model"]))
    except:
        print("fail to make QKV")
        pass

def makesuf(args,params_tr,params_sexp):
    pname="".join([f"{k}_{v}_" for k,v in params_tr.items()])
    pname=pname+"".join([f"{k}_{v}_" for k,v in params_sexp.items()])+f"_epoch{args.epochs}"
    for k,v in {"sexpfilename__":"","want_":"","num_free_vars":"var","num_layer":"l","d_model":"d_",
                "seq_len":"seq","max_depth":"depth","batch_size":"b"}.items():
        pname=pname.replace(k,v)
    if(params_tr["recursive"]):
        pname+="_recur"
    if(params_tr["attentiononly"]):
        pname+="_ato"
    if(params_tr["noembedded"]):    
        pname+="_noemb"
    return pname

def pipeline(args,
             params_sexp:dict,
             params_tr: dict ={"d_model":256, "nhead":8, "num_layer" : 4, "dim_ff": 1024, "max_len": 4096,"dropout":0.2},
             out_root="result", seed: int =-1):
    if(args.debug):
        args.n_sexps=10
    if args.show_msg:
        print("[1/5] Generating S-expressions...")
    S,ss,steps=genSexps(args,out_root)
    if args.show_msg:
        print("[3/5] Converting to Dyck language...")
    pairs,vocab_size,params_tr["seq_len"]=convert(S,ss,args)
    params_tr["max_len"]=min(args.max_len,max([len(s[0]) for s in pairs]))
    if args.show_msg:
        print("[4/5] K-fold training/evaluation...")
    pname=makesuf(args,params_tr,params_sexp)
    folds = kfold_split(len(pairs), args.kfold, args.seed)[:1]
    with open(f"log/{pname}.log","w") as fpw:
        for k, (tr_idx, va_idx) in enumerate(folds):
            os.makedirs(f"{out_root}/fold_{k+1:02d}", exist_ok=True)
            model=make_model(params_tr,args.model,vocab_size,args.debug).to(args.device)
            assert(np.any(pairs!=0))
            for i in range(len(pairs)):
                assert(np.any(pairs[i,2:4,:]>0)),print(pairs[i])
            ds_train = tensor([pairs[i] for i in tr_idx])
            ds_val   = tensor([pairs[i] for i in va_idx])
#            ds_train = [np.array(list(t)) for t in zip(*[pairs[i] for i in tr_idx])]
#            ds_val   = [np.array(list(t)) for t in zip(*[pairs[i] for i in va_idx])]            
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
                    model,train_loss,best_val_loss,last_val_loss=train_one_fold(args,model, ds_train, ds_val,
                                                            epochs=args.epochs, batch_size=args.batch_size,
                                                            device=args.device,use_amp=args.use_amp,evalperi=args.evalperi,debug=args.debug)
                    save(model.state_dict(), modelname)
                if args.show_msg:
                    msg=f"[5/5][fold {k+1}/{args.kfold}] train loss: {train_loss}, best val loss: {best_val_loss}, last val loss: {last_val_loss}"
                    print(msg)
                    print(msg,file=fpw)
                    print(f"[5/5][fold {k+1}/{args.kfold}] visualizing attentions")
                for i in range(args.n_eval):
                    eval_show(args,params_tr,model,out_root,i,vocab_size,pname,k)
                print("Fin.")

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

def test_attention(args):
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
                print("start params",args)
                pipeline_arg(args)
                print("success",args)

def  pipeline_arg(args):
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    args.use_amp=args.use_amp and (args.device=="cuda")       

    params_sexp:dict={"num":args.n_sexps,"num_free_vars":args.n_free_vars,"max_depth":args.max_depth,"sexpfilename":args.sexpfilename,"want_kind":args.want_kind}
    if(args.sexpfilename!=""):
        params_sexp["sexpfilename"]=args.sexpfilename
    params_tr: dict ={"d_model":args.d_model, "nhead":args.nhead, "num_layer" : args.num_layer, 
                        "dim_ff": args.dim_ff, "max_len": args.max_len,"dropout":args.dropout,
                        "model":args.model,"recursive":args.recursive,"attentiononly":args.attentiononly,
                        "batch_size":args.batch_size,"noembedded":args.noembedded}
    if(args.debug):
        print("start ",args)
    pipeline(args, params_sexp,params_tr,out_root=out_root)

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    args.use_amp=args.use_amp and (args.device=="cuda")       
    if(args.all):
        run_all(args,out_root)
    elif(args.simple):
        run_small(args,out_root,["simple"])
    elif(args.small):
        run_small(args,out_root)
    elif(args.test_attention):
        test_attention(args)
    else:
        pipeline_arg(args)
