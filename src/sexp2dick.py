from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
from sexpdata import Symbol, dumps, loads
import ast
import re

# Dyck 木ノード（内部表現）
@dataclass
class _Node:
    children: List["_Node"]

# =========================
# S式 <-> (Dyck, labels)
# =========================
def sexp_to_dyck_and_labels(sexp: Any) -> Tuple[str, List[str]]:
    """
    sexpdata 互換のS式（list / Symbol / int / float）を
    Dyck括弧列（"("と")"のみ）とラベル列（プレオーダ）にエンコード。
    """
    dyck_chars: List[str] = []
    labels: List[str] = []

    def visit(x: Any):
        dyck_chars.append("(")
        if isinstance(x, list):
            labels.append("LIST")
            for c in x:
                visit(c)
        elif isinstance(x, Symbol):
            labels.append(f"SYM:{str(x)}")
        elif isinstance(x, bool):
            labels.append(f"BOOL:{x}")
        elif isinstance(x, (int, float)):
            labels.append(f"NUM:{repr(x)}")
        elif x is None:
            labels.append("NONE")
        else:
            # 必要なら型を拡張（文字列等）:
            # ここでは安全のため明示エラー
            raise TypeError(f"Unsupported atom type: {type(x)}")
        dyck_chars.append(")")

    visit(sexp)
    return "".join(dyck_chars), labels


def dyck_and_labels_to_sexp(dyck: str, labels: List[str]) -> Any:
    """
    Dyck括弧列とラベル列（プレオーダ）から元のS式を復元。
    - dyck は "(" と ")" のみから成ること（空白可）
    - labels はノード数と同数
    """
    # 1) 括弧列 → 木
    cleaned = re.sub(r"\s+", "", dyck)
    if not cleaned or set(cleaned) - set("()"):
        raise ValueError("Dyck string must contain only '(' and ')' (and optional spaces).")

    stack: List[_Node] = []
    root: Optional[_Node] = None
    for ch in cleaned:
        if ch == "(":
            node = _Node(children=[])
            if stack:
                stack[-1].children.append(node)
            stack.append(node)
        else:  # ')'
            if not stack:
                raise ValueError("Unbalanced Dyck string: extra ')'.")
            node = stack.pop()
            if not stack:
                if root is not None:
                    raise ValueError("Dyck string encodes multiple roots.")
                root = node
    if stack:
        raise ValueError("Unbalanced Dyck string: missing ')'.")
    if root is None:
        raise ValueError("Empty Dyck string.")

    # 2) プレオーダでラベルをノードに対応させつつ S式を構成
    it = iter(labels)

    def assign_and_build(n: _Node) -> Any:
        try:
            lab = next(it)
        except StopIteration:
            raise ValueError("Label list shorter than number of nodes.")
        if lab == "LIST":
            return [assign_and_build(c) for c in n.children]
        elif lab.startswith("SYM:"):
            if n.children:
                raise ValueError("Symbol node must be a leaf in this encoding.")
            return Symbol(lab[4:])
        elif lab.startswith("NUM:"):
            if n.children:
                raise ValueError("Number node must be a leaf in this encoding.")
            # ast.literal_eval で int/float を安全に復元
            return ast.literal_eval(lab[4:])
        elif lab.startswith("BOOL:"):
            if n.children:
                raise ValueError("Bool node must be a leaf in this encoding.")
            val = lab[5:]
            if val == "True":  return True
            if val == "False": return False
            raise ValueError(f"Invalid BOOL label: {lab}")
        elif lab == "NONE":
            if n.children:
                raise ValueError("None node must be a leaf in this encoding.")
            return None
        else:
            raise ValueError(f"Unknown label: {lab}")

    sexp = assign_and_build(root)

    # 余剰ラベルがないことを確認
    try:
        next(it)
        raise ValueError("Label list longer than number of nodes.")
    except StopIteration:
        pass

    return sexp

# =========================
# 文字列I/Oのユーティリティ
# =========================

def sexp_str_to_dyck_and_labels(sexp_str: str) -> Tuple[str, List[str]]:
    """
    S式文字列（例: '(+ 1 (* 2 3))'）をパースして Dyck+labels に。
    """
    sexp = loads(sexp_str)
    return sexp_to_dyck_and_labels(sexp)

def dyck_and_labels_to_sexp_str(dyck: str, labels: List[str]) -> str:
    """
    Dyck+labels から S式文字列に復元。
    """
    sexp = dyck_and_labels_to_sexp(dyck, labels)
    return dumps(sexp)

# =========================
# 動作デモ
# =========================
if __name__ == "__main__":
    examples = [
        "(+ 1 (* 2 3))",
        "(if (< 1 2) (pow 3 2) (min (list 7 8 9)))",
        "(sum (list (abs -4) (round 3.14)))",
    ]
    for s in examples:
        dyck, labs = sexp_str_to_dyck_and_labels(s)
        back = dyck_and_labels_to_sexp_str(dyck, labs)
        print("S   :", s)
        print("Dyck:", dyck)
        print("Labs:", labs)
        print("Back:", back)
        print("-" * 60)
