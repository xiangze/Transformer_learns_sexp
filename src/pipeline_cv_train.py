"""
- 処理:
  S式の生成(S) → 評価(ss) → Dyck変換(D, dd) → K-fold交差検証で学習・評価
  モデルは transformer_dick_fixed_embed.py（--model fixed）
           または Recursive_Transformere.py（--model recursive）
  学習後に matrix_visualizer.py があればAttention行列を保存（--visualize）
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple
import util
import generate_sexp_with_variable as gen_mod
import evallisp as evl_mod
import sexp2dick as s2d_mod
import matrix_visualizer as vis

# ------------------------------
# Data pipeline components
# ------------------------------

def generate_S(n_sexps: int, n_free_vars: int, seed: int) -> List[List[str]]:
    """
    Call a generator in generate_sexp_with_variable.py
    Expected return: List[List[str]] (a list of tokenized S-expressions per sample)
    """
    random.seed(seed)
    gen_fn = util.smart_getattr(gen_mod, [
        "generate_dataset",
        "generate_sexp_list",
        "generate_sexps",
        "generate",
        "main",
    ])
    if gen_fn is None:
        util.fail_with_attributes(gen_mod, "S-expression generation")

    # Try common signatures
    for try_kwargs in (
        dict(n_samples=n_sexps, n_free_vars=n_free_vars, seed=seed),
        dict(n=n_sexps, n_free_vars=n_free_vars, seed=seed),
        dict(count=n_sexps, free_vars=n_free_vars, seed=seed),
        dict(n_sexps=n_sexps, n_free_vars=n_free_vars, seed=seed),
        dict(n_samples=n_sexps, n_vars=n_free_vars, seed=seed),
    ):
        try:
            S = gen_fn(**try_kwargs)
            if S is not None:
                return S
        except TypeError:
            pass

    # Positional fallback
    try:
        S = gen_fn(n_sexps, n_free_vars, seed)
        if S is not None:
            return S
    except TypeError:
        pass

    raise RuntimeError(
        "Generator function found but could not be called with expected arguments."
    )

def eval_S(S: List[List[str]], log_steps: bool) -> Tuple[List[List[str]], Optional[List[int]]]:
    """
    Evaluate each S-expression to its result tokens (ss).
    If steps-aware API exists, also return steps per sample.
    """
    eval_with_steps = smart_getattr(evl_mod, [
        "eval_list_with_steps",
        "evaluate_list_with_steps",
        "eval_with_steps",
    ])
    if eval_with_steps is not None:
        try:
            ss, steps = eval_with_steps(S)
            return ss, (steps if log_steps else None)
        except Exception:
            pass

    eval_list = smart_getattr(evl_mod, [
        "eval_list",
        "evaluate_list",
        "evalall",
        "evaluate",
        "main",
    ])
    if eval_list is None:
        fail_with_attributes(evl_mod, "S-expression evaluation")

    ss = eval_list(S)
    return ss, None

def sexp_to_dyck(S: List[List[str]], ss: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Convert S and evaluated ss into Dyck sequences D (inputs) and dd (labels).
    """
    batch_conv = smart_getattr(s2d_mod, [
        "sexp_list_to_dyck",
        "convert_list",
        "to_dyck_list",
        "convert_batch",
    ])
    if batch_conv is not None:
        try:
            out = batch_conv(S, ss)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            if isinstance(out, dict) and "D" in out and "dd" in out:
                return out["D"], out["dd"]
        except Exception:
            pass

    one_conv = smart_getattr(s2d_mod, [
        "sexp_to_dyck",
        "convert",
        "to_dyck",
    ])
    if one_conv is None:
        fail_with_attributes(s2d_mod, "S→Dyck conversion")

    D, dd = [], []
    for s, y in zip(S, ss):
        out = one_conv(s, y)
        if isinstance(out, tuple) and len(out) == 2:
            D.append(out[0]); dd.append(out[1])
        elif isinstance(out, dict) and "D" in out and "dd" in out:
            D.append(out["D"]); dd.append(out["dd"])
        else:
            raise RuntimeError("Single-item converter must return (D, dd) or {'D','dd'}.")
    return D, dd


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
                   train_pairs: List[Tuple[str, str]],
                   val_pairs: List[Tuple[str, str]],
                   out_dir: Path,
                   epochs: int,
                   batch_size: int,
                   seed: int) -> Optional[Any]:
    """
    Import the selected model module and call its train function.
    Expected callable names: train, fit, or main (CLI-like).
    When only main exists, pass jsonl paths via env vars.
    """
    if model_kind == "fixed":
        model_mod_name = "transformer_dick_fixed_embed"
    elif model_kind == "recursive":
        model_mod_name = "Recursive_Transformere"
    else:
        raise ValueError("model_kind must be 'fixed' or 'recursive'.")

    model_mod = __import__(model_mod_name)

    train_fn = smart_getattr(model_mod, ["train", "fit"])
    if train_fn is not None:
        return train_fn(train_pairs=train_pairs,
                        val_pairs=val_pairs,
                        out_dir=str(out_dir),
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed)

    main_fn = smart_getattr(model_mod, ["main"])
    if main_fn is None:
        fail_with_attributes(model_mod, "model training entrypoint (train/fit/main)")

    # Serialize jsonl files and pass paths through environment variables.
    tmp_train = out_dir / "train.jsonl"
    tmp_val = out_dir / "val.jsonl"
    save_pairs_jsonl(train_pairs, tmp_train)
    save_pairs_jsonl(val_pairs, tmp_val)

    return main_fn()

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="S→eval→Dyck→K-foldでTransformer学習・可視化まで一括実行")
    parser.add_argument("--n-sexps", type=int, default=5000, help="生成するS式サンプル数")
    parser.add_argument("--n-free-vars", type=int, default=2, help="各S式の自由変数の数")
    parser.add_argument("--kfold", type=int, default=5, help="交差検証のfold数")
    parser.add_argument("--model", type=str, choices=["fixed", "recursive"], default="fixed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./runs/exp")
    parser.add_argument("--log-eval-steps", action="store_true",
                        help="evallistがステップ数を返すAPIを持つ場合にCSV保存")
    parser.add_argument("--visualize", action="store_true",
                        help="学習後にmatrix_visualizerでAttention Matrixを保存 (可能な場合)")

    args = parser.parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] Generating S-expressions...")
    t0 = time.time()
    S = generate_S(args.n_sexps, args.n_free_vars, args.seed)
    print(f"  generated: {len(S)} samples in {time.time()-t0:.2f}s")

    print("[2/5] Evaluating S-expressions...")
    t0 = time.time()
    ss, steps = eval_S(S, args.log_eval_steps)
    print(f"  evaluated: {len(ss)} samples in {time.time()-t0:.2f}s")

    if args.log_eval_steps and steps is not None:
        step_log_path = out_root / "eval_steps.csv"
        with step_log_path.open("w", encoding="utf-8") as f:
            f.write("index,steps\n")
            for i, st in enumerate(steps):
                f.write(f"{i},{st}\n")
        print(f"  step counts saved to: {step_log_path}")

    print("[3/5] Converting to Dyck language...")
    t0 = time.time()
    D, dd = sexp_to_dyck(S, ss)
    pairs = make_pairs(D, dd)
    print(f"  converted: {len(pairs)} pairs in {time.time()-t0:.2f}s")

    print("[4/5] K-fold training/evaluation...")
    folds = kfold_split(len(pairs), args.kfold, args.seed)
    for k, (tr_idx, va_idx) in enumerate(folds):
        fold_dir = out_root / f"fold_{k+1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_pairs = [pairs[i] for i in tr_idx]
        val_pairs   = [pairs[i] for i in va_idx]

        save_pairs_jsonl(train_pairs, fold_dir / "train.jsonl")
        save_pairs_jsonl(val_pairs, fold_dir / "val.jsonl")

        print(f"  [fold {k+1}/{args.kfold}] train={len(train_pairs)} val={len(val_pairs)}")
        _ = train_one_fold(args.model, train_pairs, val_pairs, fold_dir,
                        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

        if args.visualize:
            print(f"  [fold {k+1}] visualizing attention (if supported)...")
            vis.print_and_dump_attention_params(args.model,)
            visualize_attention_if_possible(fold_dir)

        print("[5/5] Done.")


