import re
import uuid

def find_double_quotes(xs):
    return [ i for i, x in enumerate(xs) if x == '"' ]

def replace_strings(xs: str):
    idxs = find_double_quotes(xs)

    if len(idxs) % 2 != 0:
        raise RuntimeError(f"Unpaired string literal.")

    ret = []
    table = {}

    def get_unique_id():
        while True:
            uid = f"{{{str(uuid.uuid4()).split('-')[-1]}}}"
            if uid not in table:
                break
        return uid

    def prohibit_curly_braces(xs: str):
        if re.search('[\\{\\}]', xs) is not None:
            raise RuntimeError(f"curly braces are not allowed: {xs}")
        return xs

    prev_end = 0
    for begin, end in [ (idxs[2*i], idxs[2*i+1]) for i in range(len(idxs) // 2) ]:
        ret.append(prohibit_curly_braces(xs[prev_end: begin]))

        uid = get_unique_id()
        table[uid] = xs[begin: end+1]
        ret.append(uid)

        prev_end = end+1

    ret.append(prohibit_curly_braces(xs[prev_end:]))


    return ''.join(ret), table

def split_parens(xs):
    # '((hoge))' を ['(', '(', 'hoge', ')', ')'] みたいにしたい
    if re.search('\\)[^\s\\)]', xs):
        # ) の直後に ) 以外の文字が来るのは文法エラー
        raise RuntimeError(f"Needs space after ')': {xs}")

    m = re.match('^(?P<prefix>[\\(\\)]*)(?P<infix>[^\s\\(\\)]*)(?P<suffix>[\\(\\)]*)$', xs)
    if m is None:
        raise RuntimeError(f"Unknown parse failed error: {xs}.")

    prefix, infix, suffix = m.group('prefix'), m.group('infix'), m.group('suffix')

    return list(prefix) + [ infix ] + list(suffix)

def tokenize(xs: str):
    xs, table = replace_strings(xs)

    ret = []
    for x in re.split('\s', xs):
        if x != '':
            ret.extend(split_parens(x))

    def restore_strings(x, table):
        if x in table:
            return table[x]
        return x

    return [ restore_strings(x, table) for x in ret if x != '']

class Context:
    def __init__(self):
        self.stack = []

class Expression:
    def __init__(self, uid):
        self.uid = uid
        self.tokens = []

    def __repr__(self):
        return f"Expression({self.uid}, {str(self.tokens)})"

def _parse(ctx: Context, xs: list):
    table = {
        ctx.stack[-1].uid: ctx.stack[-1],
    }

    def get_unique_id():
        while True:
            uid = f"{{{str(uuid.uuid4()).split('-')[-1]}}}"
            if uid not in table:
                break
        return uid

    for x in xs:
        if x == '(':
            uid = get_unique_id()
            ctx.stack[-1].tokens.append(uid)
            # 新しいS式の開始
            exp = Expression(uid)
            ctx.stack.append(exp)
            ctx.stack[-1].tokens.append(x)
        elif x == ')':
            last_exp = ctx.stack[-1]
            last_exp.tokens.append(x)
            table[last_exp.uid] = ctx.stack.pop()

            if len(ctx.stack) == 0:
                # 閉じ括弧が多すぎる
                raise RuntimeError(f"Excessive close parenthesis: {last_exp}")
        else:
            ctx.stack[-1].tokens.append(x)

    if len(ctx.stack) != 1:
        # 閉じ括弧が少なすぎる
        raise RuntimeError(f"Unpaired parentheses: {ret}, {ctx.stack}")

    return table

def parse(xs: list):
    if len(xs) == 0:
        raise RuntimeError(f"No source code.")

    if xs[0] != '(':
        raise RuntimeError(f"Lisp must be start with '(': {xs[0]}")

    ctx = Context()
    exp = Expression('')
    ctx.stack.append(exp)

    return _parse(ctx, xs)


from io import StringIO

def writer(token, ast, operators, q, tab, indent, append_val=True):
    if token == '':
        # プログラムのエントリーポイント
        ## 全体となる関数を作成して
        q.append(f'def _f_0():\n')
        q.append(f'{tab*(indent+1)}_ret = []\n')

        ## 関数の中身を書き込んで
        for t in ast[''].tokens[1: -1]:
            writer(t, ast, operators, q, tab, indent+1)

        ## 戻り値をつけて
        q.append(f'{tab*(indent+1)}return _listeval(_ret)\n')
        ## 評価する
        q.append(f'_f_0()\n')
    elif token in ast:
        # トークンが AST にあって
        fst_token = ast[token].tokens[1]
        if fst_token not in {'let', 'define', 'if'}:
            # マクロではないなら
            ## 関数で括ってスコープを作成して
            q.append(f'{tab*indent}def _f_{indent}():\n')
            q.append(f'{tab*(indent+1)}_ret = []\n')

            ## 中身を書き込んで
            for t in ast[token].tokens[1: -1]:
                writer(t, ast, operators, q, tab, indent+1)

            ## 閉じる
            q.append(f'{tab*(indent+1)}return _listeval(_ret)\n')
            if append_val:
                ## これはつけたいときとそうでないときがあるので切り替え可能にしておく
                q.append(f'{tab*indent}_ret.append(_f_{indent}())\n')
        elif fst_token == 'let':
            # 変数の定義
            if ast[token].tokens[3] not in ast:
                # 代入する値が複雑な式でないとき
                q.append(f'{tab*indent}{ast[token].tokens[2]} = {ast[token].tokens[3]}\n')
                q.append(f'{tab*indent}_ret.append("{ast[token].tokens[2]}")\n')
            else:
                # 代入する値が複雑な式のとき
                writer(ast[token].tokens[3], ast, operators, q, tab, indent, append_val=False)
                q.append(f'{tab*indent}{ast[token].tokens[2]} = _f_{indent}()\n')
                q.append(f'{tab*indent}_ret.append("{ast[token].tokens[2]}")\n')
        elif fst_token == 'define':
            # 関数の定義
            params = ','.join(ast[ast[token].tokens[3]].tokens[1:-1])
            q.append(f'{tab*indent}def {ast[token].tokens[2]}({params}):\n')
            q.append(f'{tab*(indent+1)}_ret = []\n')
            writer(ast[token].tokens[4], ast, operators, q, tab, indent+1, append_val=False)
            q.append(f'{tab*(indent+1)}return _f_{indent+1}()\n')
            q.append(f'{tab*indent}_ret.append("{ast[token].tokens[2]}")\n')
        elif fst_token == 'if':
            # if 文
            writer(ast[token].tokens[2], ast, operators, q, tab, indent, append_val=False)
            q.append(f'{tab*indent}_ifcond = _f_{indent}()\n')
            q.append(f'{tab*indent}if _ifcond:\n')
            writer(ast[token].tokens[3], ast, operators, q, tab, indent+1, append_val=False)
            q.append(f'{tab*(indent+1)}_ifret = _f_{indent+1}()\n')
            q.append(f'{tab*indent}else:\n')
            writer(ast[token].tokens[4], ast, operators, q, tab, indent+1, append_val=False)
            q.append(f'{tab*(indent+1)}_ifret = _f_{indent+1}()\n')
            q.append(f'{tab*indent}_ret.append(_ifret)\n')
    else:
        # それ以外のときは演算子なら置き換え、そうでないならそのまま
        if token in operators:
            t = operators[token]
        else:
            t = token
        q.append(f'{tab*indent}_ret.append({t})\n')


def transpile(ast):
    tab = '    '
    indent = 0

    # リストを Lisp 風に評価する関数
    queue = [
        'from typing import Callable\n',
        'def _listeval(xs):\n',
        '    if len(xs) == 0:\n'
        '        return None\n'
        '    if isinstance(xs[0], Callable):\n',
        '        return xs[0](*xs[1:])\n',
        '    return xs\n',
        'def _car_(xs):\n',
        '    return xs[0]\n',
        'def _cdr_(xs):\n',
        '    return xs[1:]\n',
    ]

    # Lisp 演算子を Python 関数に置き換えるための辞書
    operators = {
        '+': '_add_',
        '-': '_sub_',
        '*': '_mul_',
        '/': '_div_',
        '%': '_mod_',
        '==': '_eq_',
        '!=': '_neq_',
        '>': '_gt_',
        '<': '_lt_',
        '>=': '_ge_',
        '<=': '_le_',
    }

    # 演算子置き換え用の関数を生成
    for op, fn in operators.items():
        queue.append(f'def {fn}(x, y):\n')
        queue.append(f'    return x {op} y\n')

    # 演算子以外の特殊関数を定義
    operators.update({
        'car': '_car_',
        'cdr': '_cdr_',
    })

    # 全体が複数のS式から成る場合、全体を括ってひとつのS式にする
    root = ast['']
    if root.tokens[0] != '(':
        root.tokens = ['('] + root.tokens + [')']

    # エントリーポイントから書き込み
    writer('', ast, operators, queue, tab, indent)

    # 出力
    with StringIO() as f:
        for line in queue:
            f.write(line)
        code = f.getvalue()
    return code


class Lisp:
    def __init__(self, code: str):
        self.code = code

    def transpile(self):
        tokens = tokenize(self.code)
        ast = parse(tokens)
        return transpile(ast)
    
def execSexp(sexp:str):    
    return exec(lisp.transpile())    

if __name__ =="__main__":
    lisp = Lisp("""
    (
        (define fizzbuzz (x)
            (if (== (% x 15) 0) (print "FizzBuzz")
                (if (== (% x 3) 0) (print "Fizz")
                    (if (== (% x 5) 0) (print "Buzz")
                        (print x)
                    )
                )
            )
        )
        (define FizzBuzz (x)
            (if (== x []) ()
                (
                    (fizzbuzz (car x))
                    (FizzBuzz (cdr x))
                )
            )
        )
        (FizzBuzz (list (range 1 31)))
    )
    """)        
    exec(lisp.transpile())