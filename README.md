# S式(Dick_k language)とその計算結果を学習するTransformer

## ファイル構成と使い方

src/ ソースコード

doc/ 関連文書

### 学習と評価
- S式データ生成、評価、ステップ数計算
- Dyck言語への変換、Dyck言語のベクトル化
- Transformerでの学習、評価
の全機能が入ったもの
```
  PYTHONPATH=/path/to/Transformer_learns_sexp/src \
  python pipeline_cv_train.py \
      --n-sexps 2000 \
      --n-free-vars 2 \
      --kfold 5 \
      --model fixed \
      --epochs 5 \
      --batch-size 64 \
      --output-dir ./runs/sexp2000_k5_fixed \
      --log-eval-steps \
      --visualize
```
オプション
```
#S式関連
--n_sexps, default=5000, 生成するS式サンプル数
--n_free_vars, default=2, 各S式の自由変数の数
--max_depth, default=10, 各S式の最大深さ
--sexpfilename, default="",S式をファイルから読み込む

#学習関連
--max_data_num, default=0 学習に使う最大データ数、default=0の場合は全データを使う
--kfold, default=5, 交差検証のfold数
--model, "fixed"は普通のTransformer, "recursive"は全層が同じパラメーターのRNN風、 "attentiononly"はAttention層のみのネットワーク
--epochs,     default=100
--batch_size, default=64
--seed,       default=42　乱数シード
--output_dir, default="./runs/exp"　データ出力ディレクトリ
--visualize", action="store_true",学習後にmatrix_visualizerでAttention Matrixを保存 (可能な場合)

#Transformer関連
--d_model, default=256 
--nhead, default=8 Multi Head数
--num_layer, default=4 層数
--dim_ff, default=1024
--max_len, default=4096
#その他
--debug", 
--use_s2d", 古いS式Dyck変換関数を使う
--use_gensexp", action="store_true",古いS式生成関数を使う

```

pipeline_cv_trainは
- S式生成、評価対の作成　genSexps
- Dyck language(IDで表されたtocken列)への変換 convert
- Tarnsformerによる学習、評価 train_one_fold

の3つの部分関数で構成されている。各々で以下のファイルを呼んでいる。

### モデル
- transformer_dick_fixed_embed.py  Dyck言語を入力とするTransformer
- Recursive_Transformere.py RNN風Transformer
- attentsiononly.py Attention層のみから構成されたネットワーク

### S式、Dyck言語生成
- mysexp2dick.py S式からDyck言語(IDで表されたtocken列)への変換
- randomhof_with_weight.py 高階関数S式生成、評価
### その他
- util.py　学習、評価用関数
- matrix_visualizer.py 行列可視化機能

## 評価における注意点
- 層数、S式の長さ(context長), S式の深さ、自由変数の数に対する汎化誤差の変化
- 学習後のAttention matrixが当初の予想通り単純なものになっているか(特にS式が浅い場合)
- MLPだけでは学習がうまくいかないのはmagic number±7の影響が考えられるが、逆にMulti head Attention+固定の組み替え層のみでは学習可能か？

## 先行研究
### Dyck_k decoderの構成
- [Theoretical Limitations of Self-Attention in Neural Sequence Models](https://arxiv.org/abs/1906.06755)
- [Self-Attention Networks Can Process Bounded Hierarchical Languages](https://www.arxiv.org/abs/2105.11115)によると２層のTransformerはstackを持ったマシンと等価であり第1層で括弧の数え上げ、第2層で括弧の対応の判定が行われる構成が示されている。
- [Theoretical Analysis of Hierarchical Language Recognition and Generation by Transformers without Positional Encoding](https://arxiv.org/abs/2410.12413)
### lambda計算の学習
- [Towards a Neural Lambda Calculus: Neurosymbolic AI Applied to the Foundations of Functional Programming](https://arxiv.org/abs/2304.09276)
- [Towards Neural Functional Program Evaluation](https://arxiv.org/abs/2112.04630)
### Transformerによる階層的言語の処理([日本語解説](https://joisino.hatenablog.com/entry/physics))
- [Physics of Language Models: Part 1, Learning Hierarchical Language Structures](https://www.arxiv.org/abs/2305.13673v4) 動的計画法をしている
- [Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems](https://arxiv.org/abs/2408.16293)
- [Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://arxiv.org/abs/2407.20311)
- [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405)
### その他
- [Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks](https://arxiv.org/abs/2406.02550)
有限体上の計算タスクを解かせる設定と学習ダイナミクス(grokking)に着目している点が参考になる

