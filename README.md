# S式(Dick_k language)とその計算結果を学習するTransformer

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
