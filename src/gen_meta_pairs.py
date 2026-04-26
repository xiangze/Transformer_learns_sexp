"""
「メタ関数 (map / filter / reduce / compose) を使った
S 式の簡約ステップがなるべく多くなるような式を生成・選択して、
S 式とその簡約式の対のリストを出力する機能」を追加したもの。

このファイル単体で動く CLI として書いてあり、

    python gen_meta_pairs.py --n 30 --max_depth 2 --candidates 16 \
        --kind heavy --out pairs.tsv --show

のように使う。
"""
import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import randhof_with_weight as M
from randhof_with_weight  import (
    gen_meta_simple,
    gen_meta_heavy,
    gen_and_select_high_steps,
    show_pair_summary,
    dump_pairs,
)

def make_gen_fn(kind: str):
    """
    kind:
      heavy : map/filter/reduce/compose を入れ子で深く積む (推奨)
      meta  : 単段 map/filter/reduce
      orig  : 元コードの random_typed_sexp (want_kind=any)
    """
    if kind == "heavy":
        def _g(depth, **_kw):
            return gen_meta_heavy(depth, M.OPS, M.CMPS)
        return _g
    elif kind == "meta":
        def _g(depth, **_kw):
            return gen_meta_simple(
                M.gen_list_literal_simple,
                M.gen_terminal_int0,
                M.OPS,
                depth,
            )
        return _g
    elif kind == "orig":
        def _g(depth, **_kw):
            return M.random_typed_sexp(max_depth=depth, want_kind="any")
        return _g
    else:
        raise ValueError(f"unknown kind: {kind}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate (sexp, reduced, steps) triples, "
                    "biased toward many reduction steps via meta functions.")
    p.add_argument("--n", type=int, default=20, help="number of pairs to output")
    p.add_argument("--max_depth", type=int, default=2, help="maximum nesting depth used by the generator")
    p.add_argument("--candidates", type=int, default=16, help="how many candidates to draw per pick (higher = more steps, slower)")
    p.add_argument("--min_steps", type=int, default=2, help="reject candidates with fewer than this many steps")
    p.add_argument("--kind", type=str, default="heavy", choices=["heavy", "meta", "orig"], help="generator flavor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="output file (TAB separated). default: auto-named")
    p.add_argument("--show", action="store_true", help="print summary + top-K to stdout")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    gen_fn = make_gen_fn(args.kind)

    pairs = gen_and_select_high_steps(
        num_pairs=args.n,
        max_depth=args.max_depth,
        eval_fn=M.totaleval,
        sexpr_to_str_fn=M.sexpr_to_str,
        gen_fn=gen_fn,
        candidates_per_pick=args.candidates,
        min_steps=args.min_steps,
        seed=args.seed,
        verbose=args.verbose,
    )
    out = args.out
    if out is None:
        out = (f"sexppair_metaheavy_n{args.n}_d{args.max_depth}_kind{args.kind}_cand{args.candidates}.tsv")
    dump_pairs(pairs, out)
    print(f"wrote {len(pairs)} pairs to {out}")

    if args.show:
        show_pair_summary(pairs, top_k=args.top_k)

if __name__ == "__main__":
    main()
