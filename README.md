# S式(Dick_k language)とその計算結果を学習するTransformer

## ファイル構成と使い方

### 学習と評価
- S式データ生成
- S式評価
- Dyck言語への変換
  - Dyck言語のベクトル化
- Transformerでの学習、評価

の
全機能が入ったもの
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

特定条件
```
python3 train_allinone.py
```


### モデル
- transformer_dick_fixed_embed.py  Dyck言語を入力とするTransformer
- Recursive_Transformere.py RNN風Transformer
- matrix_visualizer.py 行列可視化機能
### S式、Dyck言語生成
- gen_sexp.py　S式生成(テスト用)
- generate_dick.py　Dyck言語生成(テスト用)
- generate_sexp_with_variable.py　(自由変数のあるS式生成)
- evallist.py S式の評価
- step_counter.py S式の評価ステップ数計測
- sexp2dick.py S式からDyck言語への変換
- peval_pure.py S式の部分的評価
- random_hof_sexpr.py 高階関数生成

### その他
- train_eval.py 学習、評価の本体
- util.py　便利関数

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



## S式、Dyck言語生成

https://github.com/xiangze/Transformer_learns_sexp/tree/master/src　のgenerate_sexp_with_variable.pyでS式のリストのリストを生成し、それをSとしevallist.pyでSの各要素を評価しssに代入、sexp2dick.pyでS,ssの各要素をS式からDyck言語への変換してD,ddとしてそれらをデータ、ラベルの組としてcross validationでオプションで切り替えたtransformer_dick_fixed_embed.pyまたはRecursive_Transformere.py RNN風Transformerで学習、評価し、matrix_visualizer.pyを使って学習後のAttention Matrixを可視化するスクリプトを書いてください。S式の数、各S式自由変数の数、Transformerの種類はオプションで切り替えられるようにしてください。またオプションでevallist.pyでSの各要素を評価終了までに要したステップ数をログとして保存してください

ニューラルネット(DNN)の学習ではSGDに比べは徐々にランダムウォークの分散を減らしていくシミュレーテッドアニーリング的な方法は性能が北内と言われています。またDNNでは局所最適解が学習によって大域最適解に到達するのではないかとも言われています。一方スピングラスモデルでは無数の局所最適解が存在し、温度によって各ポテンシャルの状態の分配関数における割合が変化していくと言われています。このDNNとスピングラスモデルの局所解の性質の違いがシミュレーテッドアニーリングの有効性の違いに関係しているのでしょうか？あるいはシミュレーテッドアニーリングに対する挙動の違いから多数の局所解の性質の説明ができるのでしょうか。
また拡散モデルの生成に温度とポテンシャルを徐々に下げていくシミュレーテッドアニーリング的な方法が使われることも勘案して既存研究の結果に基づいて説明してみてください。

Pytorchでバッチサイズ×学習率、ラベルノイズ比率、SWA/SGDRの有無の組み合わせをを掃引してResNet18,Resnet50その他Pytrochの学習済みネットワークに対してテスト誤差、平坦度の計算結果を出力、記録するスクリプトを書いてください
