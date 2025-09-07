# S式(Dick_k language)とその計算結果を学習するTransformer

## ファイル構成と使い方
# #学習と評価
データ生成、S式評価、学習全機能が入ったもの
```
python3 train_allinone.py
```
より一般化された条件用
```
python3 train.py
```
## モデル
- transformer_dick_fixed_embed.py  Dyck言語を入力とするTransformer
- Recursive_Transformere.py RNN風Transformer
- matrix_visualizer.py 行列可視化機能
## S式、Dyck言語生成
- generate_dick.py　Dyck言語生成(テスト用)
- generate_sexp_with_variable.py　(自由変数のあるS式生成)
- evallist.py S式の評価
- sexp2dick.py S式からDyck言語への変換




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
