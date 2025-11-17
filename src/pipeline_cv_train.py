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
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import util 
import randhof_with_weight as hof

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y in pairs:
            f.write(json.dumps({"input": x, "label": y}, ensure_ascii=False) + "\n")

# ------------------------------
# Training & visualization
# ------------------------------
import transformer_dick_fixed_embed as fixed
import Recursive_Ttansformer as recursive
def train_one_fold(model_kind: str,
                   ds_train,
                   ds_val,
                   train_pairs: List[Tuple[str, str]],
                   val_pairs: List[Tuple[str, str]],
                   out_dir: Path,
                   epochs: int,
                   batch_size: int,
                   seed: int) -> Optional[Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    pin = (device == "cuda")
    
    if model_kind == "fixed":
        model=fixed.TransformerRegressor()
    elif model_kind == "recursive":
        model=recursive.SharedTransformerRegressor()
    else:
        raise ValueError("model_kind must be 'fixed' or 'recursive'.")

    num_workers=1
    train_loader = DataLoader(ds_train[:], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(ds_val[:], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    criterion=nn.MSEloss() #soft
    opt=optim.Adam(model.parameters(), lr=0.05)
    scheduler=optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    best_val_acc,last_val_acc=util.traineval(epochs,device,model,train_loader,val_loader,criterion,opt,scheduler,use_amp=True,scaler=scaler,eval=True)
    
    # Serialize jsonl files and pass paths through environment variables.
    tmp_train = out_dir+ "/train.jsonl"
    tmp_val = out_dir+  "/val.jsonl"
    save_pairs_jsonl(train_pairs, tmp_train)
    save_pairs_jsonl(val_pairs, tmp_val)
    return best_val_acc,last_val_acc

def pipeline(args,out_root="result", seed: int =-1):
    if(args.debug):
        args.n_sexps=10
    print("[1/5] Generating S-expressions...")
    t0 = time.time()
    if(args.sexpfilename!=""):
        with open(args.sexpfilename,"r",encoding="utf-8") as f:
            S=[ line.strip() for line in f.readlines()]
        ss=[r[1] for r in S]
        steps=[r[2] for r in  S]
        S=[r[0] for r in  S]
        print(f"  loaded: {len(S)} samples in {time.time()-t0:.2f}s")
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
        results=hof.gen_and_eval(args.n_sexp,args.max_depth,seed=seed)
        ss=[r[1] for r in S]
        steps=[r[2] for r in  S]
        S=[r[0] for r in  S]

    if args.log_eval_steps and steps is not None:
        step_log_path = out_root + "/eval_steps.csv"
        with step_log_path.open("w", encoding="utf-8") as f:
            f.write("index,steps\n")
            for i, st in enumerate(steps):
                f.write(f"{i},{st}\n")
        print(f"  step counts saved to: {step_log_path}")

    print("[3/5] Converting to Dyck language...")
    t0 = time.time()
    if(args.use_myconverter):
         Dyks  = mys2d.sexp_str_to_dyck_and_labels(S) 
         ssDyks= mys2d.sexp_str_to_dyck_and_labels(ss) 
    else:
         Dyks  = s2d.sexp_str_to_dyck_and_labels(S) 
         ssDyks= s2d.sexp_str_to_dyck_and_labels(ss) 

    #pairs = make_pairs(Dyks, dlabels)
    pairs = make_pairs(Dyks, ssDyks)
    print(f"  converted: {len(pairs)} pairs in {time.time()-t0:.2f}s")

    print("[4/5] K-fold training/evaluation...")
    folds = kfold_split(len(pairs), args.kfold, args.seed)
    for k, (tr_idx, va_idx) in enumerate(folds):
        fold_dir = f"{out_root}/fold_{k+1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_pairs = [pairs[i] for i in tr_idx]
        val_pairs   = [pairs[i] for i in va_idx]

        save_pairs_jsonl(train_pairs, fold_dir + "/train.jsonl")
        save_pairs_jsonl(val_pairs, fold_dir + "/val.jsonl")

        print(f"  [fold {k+1}/{args.kfold}] train={len(train_pairs)} val={len(val_pairs)}")
    
        ds_train = fixed.ExprDataset(train_pairs, mode="dyck")
        ds_val   = fixed.ExprDataset(val_pairs,   mode="dyck")
        print("[4/5] token to Tensor...")

        _ = train_one_fold(args.model, ds_train, ds_val, fold_dir,
                        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

        if args.visualize:
            print(f"  [fold {k+1}] visualizing attention (if supported)...")
            vis.print_and_dump_attention_params(args.model,)
        print("[5/5] Done.")

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="S→eval→Dyck→K-foldでTransformer学習・可視化まで一括実行")
    parser.add_argument("--n_sexps", type=int, default=5000, help="生成するS式サンプル数")
    parser.add_argument("--n_free_vars", type=int, default=2, help="各S式の自由変数の数")
    parser.add_argument("--max_depth", type=int, default=10, help="各S式の最大深さ")
    parser.add_argument("--kfold", type=int, default=5, help="交差検証のfold数")
    parser.add_argument("--model", type=str, choices=["fixed", "recursive"], default="fixed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./runs/exp")
    parser.add_argument("--log_eval_steps", action="store_true",  help="evallistがステップ数を返すAPIを持つ場合にCSV保存")
    parser.add_argument("--visualize", action="store_true",
                        help="学習後にmatrix_visualizerでAttention Matrixを保存 (可能な場合)")
    parser.add_argument("--use_gen", action="store_true",  help="S式評価にhyの部分評価を使う")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_myconverter", action="store_true")
    parser.add_argument("--use_gensexp", action="store_true",help="use old sexp generator")
    parser.add_argument("--sexpfilename", type=str, default="",help="use sexp from file")
    args = parser.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pipeline(args,out_root)
