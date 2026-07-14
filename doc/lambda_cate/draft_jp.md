# Transformers as Functional Dynamics, equivalency between lambda calculas and linear logic

## Abstract
本論文で我々は関数マップとself-attentionとの等価性をまず示し、パラメーターεをもつ線形補間型関数マップが圏論の米田の補題から関数マップ間の関手に等しいことを示す。
Transfromer,特にその一部のself-attentionが持っているとされる文脈内学習、高階論理計算に必要な機能が関数マップとの等価性の考察から冪を持つ圏、対象モノイダル閉圏(SMCC)を用いて説明できることを主張する。一方で通常λ計算を表現するための圏であるデカルト閉圏(CCC)とは等しくなく、λ計算の機能を持つが、１つの命題や数量は一回しか論理演算に使えないという線形論理(線形λ計算)に従うことを説明する。Transfromerの別の要素である残差接続が前の層から計算以前のデータを"コピーする"ことで線形論理の制約は緩和され、計算能力がSMCCの枠を超えて拡張されることを主張する。
さらにMLP(Multi layer perceptron)には
Attention matrix にsoftmaxが適用されることから確率分布関数を対象とし、その間のMarkov kernelのような変換を射としたMarkov圏でモデル化することが可能である。
結論としてTransformerはMarkov圏とSMCCの組み合わせ、さらに非線形関数変換でその性質が拡張されうることを主張する。
ここまではTransformerの推論、生成における高階論理的性質に関する考察だが，学習可能性に関しては学習機能の力学系的な考察が必要であり、単純に学習過程を自然変換とはみなせず、そこでは射の射を対象として持つ2-圏の考えが必要であり、Cartesian reverse derivative category (CRDC)とそれによる２種類のJacobiansの分別、パラメーターに対するJacobian $R[f]$と状態に対する$D_A[f]$を分けて考えるべきだがそれらは連鎖微分則により直接結びついていること、リャプノフスペクトラムがネットワークの同力学を支配していること、しかしgrokkinのような学習ダイナミクスとリャプノフスペクトルが関連してはいるが直接ではない。最後に力学系における分岐理論が力学系としてのネットワークののスペクトルを記述するデカルト逆微分圏(Cartesian reverse derivative category ,CRDC)
 によってなぜgrokkingのような学習ダイナミクスとリャプノフスペクトルが関係するのかを明らかにする。最後に我々は微分構造、スペクトル、位相などが付加された構造としてのCRDC内での力学系の分岐が圏論の基礎的な語彙、関手や自然変換に回収されることと、そのような試みの紹介をする。
  
*Keywords:* category theory, Transformers, sefl-attention, function vectors, function dynamical, dynamica systems, linear λ-calculus, Markov categories, reverse derivative categories, tangent categories
----

## Introduction
この論文ではTransformerのin context learningにおける計算能力をλ計算、圏論との関係から理論的に説明できないかと試みています。
[デカルト閉圏(CCC)](https://ja.wikipedia.org/wiki/%E5%86%AA%E5%AF%BE%E8%B1%A1)という圏は2つの対象の直積と冪対象(指数対象 )という対象およびそこへの射を持つものとして定義され、図のように対象X,Y,Zと射に対して自然変換

$Hom(X\times Y,X)\simeq Hom(X,Z^Y)$が成り立つようなものをいいます。
$X→Z^Y$射がλ計算でいうカリー化、$Z^Y→Z$射がλ計算、プログラミングでいうS式の評価(eval)に相当します。evalがAttention matrixをQ,Kから作る操作、関数マップ[]で言うf○fに相当します。

しかしベクトル空間を対象、線形写像を射とするような圏Vectは積がCCCとはならず、より制限のある対称モノイド閉圏(SMCC)という圏にしかなりません。そこでは普通の関数を対象としてevalするようなλ計算ができず、線形論理という命題を使えばなくなってしまう有限の資源として推論を行うような制約が課されます。しかし残差接続で前の層から値を持ってくることで使用できる命題の制約を緩和できると素朴には考えられます。

非常によく似たモチベーションの先行研究”[Topos of Transformer Networks](https://arxiv.org/abs/2403.18415v2)”ではReluを使うことを前提とし、区分線形写像を射とするような圏RELU上のニューラルネットを考えることでTransformerをCCCのより特別な場合であるトポスとみなせ、通常の形式主義論理演算、λ計算が可能であると主張しています。しかしこの論文は区分線形でない関数(射)を構成に使用している、AttentionとMLPに関する説明がない、実際にはトポスではなくpretoposであることしか言っていないなどいくつかの問題があり、前提条件と結論の差異をはっきりさせる。

先行研究[Transformer Feed-Forward Layers Are Key-Value Memories](arXiv:2012.14913)で言われているようにTransformerのMLP部分の機能は辞書を引くようなものであるのかどうかの数値実験結果を

その他Transformerの入力ベクトルx，あるいはそのQ,Kとの積の各行が確率分布を表しているとするとそれもAttention matrixを射とする確率分布関数の圏(マルコフ圏)として表され、Transformerはマルコフ圏とSMCCの組み合わせで表現できるというのがより完全な形式です。

線形論理に関しては変数や関数など使える資源に制約を持たせることでバグを出にくくするRustというプログラミング言語や特異学習理論、ベイズ推定のプログラミングと関連付ける研究　[program as singular ](https://arxiv.org/abs/2504.08075)があるなど興味深いです。

This paper almost explaing about the relation between Transformers, lambda calculation and category theory. But the original idea and motivation is dynamical system of functions, this is deeply related to learnability of transformers.

## Contributions of this paper
- Point out equivalence between self-attention and functional dynamics, property category .
- Explains the relation between self-attention and symmetric monoidal closed category (SMCC) $(\mathbf{Vect},\otimes,\multimap)$ eval-apply loop which is required for in-context learning.
- Numerical experiment about MLP function, only Key-value retirival or not.

## Preriminalies
### Transformers and theire Components
Transformers are consists of several components, attention, MLP(multi layer perceptron ,FFN), softmax, residual connecctions and layer normalizations.
Attentions are product of  and input vector x.
MLP is composition of all-to-all vector product using matrix product and activation function such as Relu or softmax.
softmax function is usually used tu make attention matrix in conrast of Relu in the tail of MLP.
Residual connection (Resnet) is often used in LLM. The benefits of Resnet are not only preserving information of earlier layers during training and inference, but simplify loss landscape. Resnet with nearly identical matrix convertion are similar to differential operations  which is called neural ordinal differencial equiation(neural ODE).
Layer normalizations are 

positional encoders are also important for identify the order of tokens.
layered transformers is usually called large language models (LLM).
this have  context learning alibity and scalability of learning.
LLMs and their variants made various applications and theoretical explanations.

Function vector(FV) [] a concept embedded in LLM as a head of transformer. FV is portable among but layer of LLM.

### Functional Dynamics
Functional dynamics (FD)[] is introduced by function of 1-dimentional function (metafunction).
The original FD is define as following

$f_{t+1}(x)=(f_t\cdot f_t)(x) +\epsilon f_t(x)$

or 

$f_{t+1}(x)=(f_t\cdot g_t)(x) +\epsilon f_t(x)$

Generally, the dynamics of functions are governed by fixed points and hieralchy of fixed points and the structure complex behavior depends on initial function f and parameter $\epsilon$[].
Regarding a fnction as a graph drawn in 2D rectangle, function applicaton to other function ($f\cdot g$) is described as matrix multiplication. In case attention mechanism, f and g corespons to matrix, the non-zero value is  
row is x -axis, column is y-axis graph.Then a matrix not only represent 1-dimentional function graph but 

f and g corresponds to morphism, the functor is functional dynamics.
In other formulation f,g are objects, functional dynamics itself is morphism and the functor is parametrize by $\epsilon$.
By restricting the formular of FD linear interpolations as in the original paper, category theory can explation  its parameters $\epsilon$.
Functor between FD and parameter $\epsilon$ is natual transformation. Yoneda's lemma $Nat(h_A,F)\simeq F(A)$ corresponds to this relation is 

### Category Theory
As described above, FD can be treated as some kind of Category.
Lambda calculus treats functions as same as variables. All calculation in is multiple steps of evaluations(eval) and applications(apply) of formulars.
Eval is so called charactor string as a formular and calucule this, apply is the process that substitutiig eval's result to other formular. This eval-apply loop is common at the various field of computer programming.
Lambda calcuals has three rules, alpha conversion beta reduction and eta conversion. Alpha conversion is just replacement of bound variable names. Beta reduction is application of a function described by $(\lambda x. f) b=f(b)$ in usual notation. Eta conversion is desciribed as $(\lambda x. f) x=f $, here rhs and lhs are same function (constant). This is corresponds to extentionality definition of functions sets theorem.

lambda calculus is formulate by using Cartesian closed category (CCC) which have product $X \times Y $ of tow objects X,Y and exponential object $X^Y$.
There is natual bjiction $Hom(X \times Y,Z) \simeq Hom(X,Z^Y)$. There is one morphism called carring $\lambda g$ for all g and 
morphism $Z^Y→Z$ is evaluation of program(S-expression in LISP),this is coresspons to calculation of Attention matrix from Q and K, $f\cdot f$ of functional dynamics.

$$\begin{CD}
A @>{f}>> B \\
@VV{\lambda g}V @VV{g}V \\
C @ C @ .
\end{CD}$$

### Linear Logic, Linear lambda calculus
Linear logic is restriction of usual mathematical logic which only allows finite use of propositions during a deduction.
Topos
## Formulation
This chapter explains calculation ability of transformer based on the idea of functional dynmics and spesific cateories composition related to  relation of linear logic.

## Experiment 
Along to FV.

## Discussion

### Related works

### Learnability

## Conclutions and Perspectives
In this paper, we show the correndence between lambda calculas and transformer, functional dynamics and explain the linear calculation ability in-context of transformers is composed Markov and SMCC categories.
### Reference

- FV
- FV
- FD
- Catefory
- Linear logic
- singular program
- Rust
- Transformer
- Topos
- Endofunctor

