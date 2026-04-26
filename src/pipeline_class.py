# pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

import util
from util import mprint, dprint

import mysexp2dick as mys2d
import matrix_visualizer as vis
import attentiononly as atn
import transformer_dick_fixed_embed as fixed
import randhof_with_weight as hof
import argparse

# ──────────────────────────────────────────────
# 1. 設定
# ──────────────────────────────────────────────
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
    epochs: int = 1000  # 学習エポック数
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
    activate: bool = False #True use gelu for Attention only NN
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
    clean:bool=False
    task:str="regression" #regression is for S-exp,classifiation is for test
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
    parser.add_argument("--model", type=str, choices=["fixed", "recursive", "attentiononly", "outQK"], default=defaults.model,)
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
    parser.add_argument("--activate", action="store_true")
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
    parser.add_argument("--clean", action="store_true") #remove cache files
    parser.add_argument("--task", type=str, default=defaults.task)
    # old
    parser.add_argument("--use_s2d", action="store_true")
    return parser

def parse_args() -> PipelineArgs:
    parser = build_parser()
    ns = parser.parse_args()
    return PipelineArgs(**vars(ns))

# ──────────────────────────────────────────────
# 2. データ管理
# ──────────────────────────────────────────────
class DataManager:
    """S式の生成・トークン化・キャッシュ・テンソル変換を担う"""
    def __init__(self, args: PipelineArgs):
        self.args = args

    # --- S式生成 ---
    def generate(self, num: int = 0, seed: int = None) -> Tuple[List, List, List]:
        args = self.args
        seed = seed or args.seed
        num = num or args.n_sexps

        if args.sexpfilename:
            return self._load_from_file(args.sexpfilename)

        mprint("[2/5] Evaluating S-expressions...", args.show_msg)
        simplekinds = ["simple", "meta", "arith", "add", "ring","heavy"]
        if args.want_kind in simplekinds:
            SS = hof.gen_and_eval_heavy(num, args.max_depth, seed=seed, want_kind=args.want_kind ,debug=args.debug)
        elif args.want_kind in simplekinds:
            SS = hof.gen_and_eval_simple(num, args.max_depth, seed=seed,
                                         want_kind=args.want_kind, n_free_vars=args.n_free_vars)
        else:
            SS = hof.gen_and_eval(num, args.max_depth, seed=seed,
                                  want_kind=args.want_kind, n_free_vars=args.n_free_vars)
        S, ss, steps = map(list, zip(*SS))
        return S, ss, steps

    def _load_from_file(self, path: str) -> Tuple[List, List, List]:
        with open(path, "r", encoding="utf-8") as f:
            ls = [line.strip().split(",") for line in f]
        S, ss, steps = map(list, zip(*ls))
        if self.args.max_data_num > 0:
            S = S[:self.args.max_data_num]
        return S, ss, steps

    # --- トークン化・キャッシュ ---
    def load_or_convert(self, src_sexps, tgt_sexps, maxlen: int = 0) -> Dict:
        cache_path = self._cache_path()
        if os.path.isfile(cache_path):
            try:
                return self._load_cache(cache_path)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[cache] 読み込み失敗 ({e})。再変換します。")
        return self._tokenize_and_save(src_sexps, tgt_sexps, cache_path, maxlen)

    def _cache_path(self) -> str:
        args = self.args
        if args.sexpfilename:
            return f"{args.sexpfilename}_conv.json"
        return (f"sexp/sexppair_n{args.n_sexps}_d{args.max_depth}"
                f"_freevar{args.n_free_vars}_kind{args.want_kind}.txt_conv.json")

    def _tokenize_and_save(self, src, tgt, path, maxlen) -> Dict:
        tokenss, _, maskss = mys2d.sexpss_to_tokens(src, tgt, show=False, maxlen=maxlen)
        data = {"srcs": tokenss[0], "targets": tokenss[1],
                "src_masks": maskss[0], "target_masks": maskss[1]}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data

    def _load_cache(self, path: str) -> Dict:
        mprint(f"[cache] loading {path}", self.args.show_msg)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: [list(map(int, r)) for r in data[k]] for k in data}

    # --- numpy配列への変換 ---
    def to_pairs(self, tokens: Dict, maxlen: int = 0) -> Tuple[np.ndarray, int, int]:
        vocab_size = max(
            [max(s) for s in tokens["srcs"]] +
            [max(s) for s in tokens["targets"]]
        ) + 1
        pairs = np.array(list(tokens.values())).transpose((1, 0, 2))
        max_seq_len = max(len(s) for s in tokens["srcs"])
        return pairs, vocab_size, max_seq_len

    @staticmethod
    def kfold_split(n: int, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        folds, fold_size = [], max(1, n // k)
        for i in range(k):
            start, end = i * fold_size, n if i == k - 1 else (i + 1) * fold_size
            folds.append((idx[:start] + idx[end:], idx[start:end]))
        return folds

# ──────────────────────────────────────────────
# 3. モデル生成
# ──────────────────────────────────────────────
class ModelFactory:
    """paramsとmodel_kindからモデルインスタンスを生成する"""

    @staticmethod
    def build(params: dict, model_kind: str, vocab_size: int,
              task: str = "regression", debug: bool = False) -> nn.Module:
        params = {**params, "pad_id": 0, "vocab_size": vocab_size}
        embedding = not params.get("noembedded", False)

        if params.get("attentiononly"):
            return ModelFactory._build_attentiononly(params, task, debug, embedding)
        else:
            return ModelFactory._build_transformer(params, model_kind, task, debug)

    @staticmethod
    def _build_attentiononly(params, task, debug, embedding):
        recursive = params.get("recursive", False)
        act = params.get("activate", False)
        if task == "classification":
            return atn.AttentionOnlyClassifier(params, debug=debug,
                                               recursive=recursive, embedding=embedding, act=act)
        if recursive:
            return atn.AttentionOnlyRecursiveRegressor(params, debug=debug,
                                                       weightvisible=True, embedding=embedding, act=act)
        return atn.AttentionOnlyRegressor(params, debug=debug, embedding=embedding, act=act)

    @staticmethod
    def _build_transformer(params, model_kind, task, debug):
        is_recursive = (model_kind == "recursive")
        out_qk = (model_kind == "outQK")
        params["recursive"] = is_recursive
        if task == "classification":
            return fixed.TransformerClassifier(params, debug=debug, outQK=out_qk)
        return fixed.TransformerRegressor(params, debug=debug, recursive=is_recursive, outQK=out_qk)

# ──────────────────────────────────────────────
# 4. 学習・評価
# ──────────────────────────────────────────────
class Trainer:
    """1fold分の学習・評価・可視化を担う"""
    def __init__(self, args: PipelineArgs, params: dict):
        self.args = args
        self.params = params

    def train_fold(self, model: nn.Module, ds_train, ds_val, fpw=None):
        args = self.args
        pin = (args.device == "cuda")
        evalperi = max(1, args.epochs // 10)
        train_loader = DataLoader(TensorDataset(ds_train), batch_size=args.batch_size,
                                  shuffle=True, pin_memory=pin)
        val_loader   = DataLoader(TensorDataset(ds_val),   batch_size=args.batch_size,
                                  shuffle=False, pin_memory=pin)

        criterion = nn.CrossEntropyLoss() if args.task == "classification" else nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=0.05)
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 0.95 ** e)

        return util.traineval(
            args.epochs, args.device, model, train_loader, val_loader,
            criterion, opt, scheduler,
            use_amp=args.use_amp, eval=True, peri=evalperi,
            debug=args.debug, fpw=fpw, task=args.task
        )

    def visualize(self, model, vocab_size, pname, k, pair):
        args = self.args
        xin, mask = self._to_tensor([pair])
        vis.save_attention_heatmap(model, self.params, vocab_size, args.device,
                                   f"{pname}_{k}", x=xin, mask=mask, out_dir="img/")

    def _to_tensor(self, pairs):
        ds = [tensor(np.array(list(t))) for t in zip(*pairs)]
        mask = torch.ones(ds[2].shape).to(self.args.device)
        return ds[0].to(self.args.device), mask


# ──────────────────────────────────────────────
# 5. パイプライン（組み合わせ）
# ──────────────────────────────────────────────
class Pipeline:
    """DataManager / ModelFactory / Trainer を束ねてK-fold学習を実行する"""

    def __init__(self, args: PipelineArgs):
        self.args = args
        self.data_mgr = DataManager(args)

    def run(self, params_tr: dict,fpw=None):
        args = self.args
        out_root = Path(args.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # データ準備
        mprint("[1/5] Generating S-expressions...", args.show_msg)
        S, ss, _ = self.data_mgr.generate()

        mprint("[3/5] Tokenizing...", args.show_msg)
        tokens = self.data_mgr.load_or_convert(S, ss, maxlen=args.max_len)
        pairs, vocab_size, max_seq_len = self.data_mgr.to_pairs(tokens)
        params_tr.update({"seq_len": max_seq_len,
                          "max_len": min(args.max_len, max_seq_len)})

        # K-fold
        mprint("[4/5] K-fold training...", args.show_msg)
        folds = DataManager.kfold_split(len(pairs), args.kfold, args.seed)[:1]
        trainer = Trainer(args, params_tr)

        for k, (tr_idx, va_idx) in enumerate(folds):
            self._run_fold(k, pairs, tr_idx, va_idx,
                            vocab_size, params_tr, trainer, fpw)

    def _run_fold(self, k, pairs, tr_idx, va_idx, vocab_size, params_tr, trainer, fpw):
        args = self.args
        model = ModelFactory.build(params_tr, args.model, vocab_size,
                                   task=args.task, debug=args.debug).to(args.device)
        ds_train = tensor(np.array([pairs[i] for i in tr_idx]))
        ds_val   = tensor(np.array([pairs[i] for i in va_idx]))
        #train_loss, best_val_loss, val_loss
        train_loss, best_val, best_val = trainer.train_fold(model, ds_train, ds_val, fpw)
        dprint(f"[fold {k+1}] train={train_loss:.4f} best_val={best_val:.4f}", fpw)

        for i in range(args.n_eval):
            trainer.visualize(model, vocab_size, f"fold{k}", k, pairs[va_idx[i]])


# ──────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    params_tr = {
        "d_model": args.d_model, "nhead": args.nhead, "num_layer": args.num_layer,
        "dim_ff": args.dim_ff, "max_len": args.max_len, "dropout": args.dropout,
        "model": args.model, "recursive": args.recursive, "attentiononly": args.attentiononly,
        "batch_size": args.batch_size, "noembedded": args.noembedded, "activate": args.activate,
    }
    with open(f"log/run_pipelineclass.log", "a") as fpw:
        Pipeline(args).run(params_tr)