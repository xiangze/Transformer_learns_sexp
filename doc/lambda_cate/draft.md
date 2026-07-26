# Transformers as Functional Dynamics, equivalency between lambda calculas and linear logic
Categoriy theoretical view of transformas and functional dynamics, threre ability of higher order calculation
## Abstract
In this papaer we show the equivalence of functional dynamics and self-attention mechanism, then linear interpolation type functional dynamics whose parameter is $\epsilon$ is equivalent to the functor between functinoal dynamics from Yoneda's lemma.
From the point of view the equivalence between attention and functional dynamics, 
In-context learning, higher order logic function ability which seems transfromers, especially the part of self-attention have can be explained by symmetric monoidal closed category (SMCC) which has exponenatioal object.
SMCC is not equivalent to Cartesian closed category (CCC), both has higher order $lamda$-calculas function but we explain SMCC obeys linear logic(linear $lambda$ calculation) which one property or value can be used only once during calculation or proof.

And we state residual connection, another component of transformers "copy" data from previous layer, the constraint of linear logic is weakend and comutation ability recovers from the original SMCC.

Moreover we show MLP(Multi layer perceptron) has a function to retrive information from key-value database experimentally.

Because softmax is applied attention matrix, we can define Markov category as the object is probablistic distribution function, the morphism is converison like Markov kernel.

The main statement of this paper is drawn as transformers are mutial adjoint between piecewise-linear, Cartesian category corresponds to MLP and linear category corresponds to attention, expecially an attention matrix is a composition of Markov category and SMCC.

This category theoretical structure expresses the type of representaiton ability enough, specific learned network parameter values also affects the performance of network but limited as evaluated by numerical experiments.

Numerical experiments show the recovery of restriction of linear logic in attention mechanism and  which part of transformer affects outstandin performance of transformer.
Related numerical experiments and theorem formulation for automatic prooves are provided https://github.com/xiangze/Transformer_learns_sexp/tree/master/src/FV and https://github.com/xiangze/Transformer_learns_sexp/tree/master/proof 

Keywords:
category theory, Transformers, sefl-attention, function vectors, function dynamical, dynamical systems, linear $\lambda$-calculus, Markov categories

----

## Introduction
Almost universal computation power of Transfomers attracta many reseachers. Especiallty universal turing machine(UTM) ,lambda calculation theory and category theory are usually use to explain them.

Cartesian closed category(CCC) is defined having objects called direct product between two objects $X\times Y$ and exponential object(ofnen written $Z^Y$) .Intuitively an exponential object$X^Y$ is set of all morphisms from X to Y.
has natural transformation $Hom(X\times Y,X)\simeq Hom(X,Z^Y)$ for objects $X,Y,Z$.

In $\lambda$ calculas or programming points of view, morphism $X→Z^Y$ is currying, $Z^Y→Z$ is $\lambda$ calculation, i.e eval of S-expression.As explained bellow eval is the oparation to generate attention matricx from matrix Q and K in transformers, function application $f\cdot f$ in functional dynamics.

But category $\bf{Vect}$ whose objects is vector space, morphisms are linear transformation can not have nonlinear diagnal product. This is not CCC but called Symmetric Monoid Closed Category(SMCC). In this categorey usual eval can not used for functions without constraint , but use linear logic deduction, which treat propositions as finite resources when it used comsumed.

In same motivation research as this paper, "Topos of Transformer Networks"(https://arxiv.org/abs/2403.18415v2)”
The assume Neural networks on category $\bf{RELU}$ whose objects are usual vector space, morphisms are partially linear map, because  Relu is usually useed as activation function. Then transformers can be treated topos, which is special case of CCC,and have univarsal higher order caluculation ability.

Linear logic is related to programming language such as Rust[] which constraint resource(such as variables) usage at one time. This reduces programming bugs. Also There is a research to connect linear logc and probablistic programming, bayesian inference [].

In this paper almost explaing about the relation between Transformers, lambda calculation and category theory. But the original idea and motivation about higher order fuction is Vector space and moprphism or category of metafunction(funcsions between funcions) is from dynamical system of functions, this is deeply related to learnability of transformers. Some theorems explained following chapter are written and proven in Lean.

### Contributions of this paper
- Points out the equivalence of self-attention and functional dynamics, property category .
- Explains the relation between self-attention and symmetric monoidal closed category (SMCC) $(\mathbf{Vect},\otimes,\multimap)$ eval-apply loop which is required for in-context learning and the correspondence between markov category and attention matrix.
- Numerical experiment about MLP function, only Key-value retirival or not.
- The proofs of formalizations are witten and proven in lean

## Preriminalies
### Attention mechanism, Transformers and their Components
Transformers are consists of several components, attention, MLP(multi layer perceptron ,FFN), softmax, residual connecctions and layer normalizations.

Fig

Attentions are product of  and input vector x.
MLP is composition of all-to-all vector product using matrix product and activation function such as Relu or softmax.
softmax function is usually used tu make attention matrix in conrast of Relu in the tail of MLP.
Residual connection (Resnet) is often used in LLM. The benefits of Resnet are not only preserving information of earlier layers during training and inference, but simplify loss landscape. Resnet with nearly identical matrix convertion are similar to differential operations  which is called neural ordinal differencial equiation(neural ODE).
Layer normalizations are another important part of transformers to regulize internal data.

![](img/summary.png)

Positional encoders are also important for identify the order of tokens which is encoded and put in attention mechanism.
Layered transformers is usually called large language models (LLM). LLMs have in-context learning ability[] and scalability of learning. LLMs and their variants made various applications and theoretical explanations.

Function vector(FV) [] a concept embedded in LLM as a head of transformer. FV is portable among but layer of LLM.

### Functional Dynamics
Functional dynamics (FD)[] is introduced by function of 1-dimentional function (metafunction). The original form FD is define as following

$f_{t+1}(x)=(f_t\cdot f_t)(x) +\epsilon f_t(x)$

or 

$f_{t+1}(x)=(f_t\cdot g_t)(x) +\epsilon f_t(x)$

Generally, the dynamics of functions are governed by fixed points and hieralchy of fixed points and the structure complex behavior depends on initial function f and parameter $\epsilon$[].
 Regarding a fnction as a graph drawn in 2D rectangle, function applicaton to other function ($f\cdot g$) is described as matrix multiplication. In case attention mechanism, f and g corespons to matrix, the non-zero value is  
row is x -axis, column is y-axis graph.Then a matrix not only represent 1-dimentional function graph but 

As a example $d=d_q=d_k=d_v$ for simplicity, x has only 1 nonzero value per one row. Then the elements of matrix looks a graph of 1 dimentional function of 2 dimentional region. When function f(red curve) is applied to f itself, one can plot $f(x)\cdot f(x)$ following f(x) value for each x coordinate. one peak function is converted to  2 peak function, 4 peak function ,8 peak ... and so on.

![](img/folding_simple.png)
Then suppose $W$ as diagonal matrix,consider product of matrix Q,K and attention map A, if the shape of elements of A is same as the ones of V, one can fold graph of Q or K by matrix product. This can be thought the product of Q and K. 
これはAの元になっているQの形状をKで引き伸ばすような処理によって得られます。

When calculating the row of Z where blue point exist, multiply product of blue row of Q and each column of K. In this case non-zero element of blue row  is only blue point, 
the column number 
As a result, only the element of Z where same row as blue row of Q, same column of blue columun of K is non-zero.

その列数(横←→)と同じ行数に非零要素があるVの列(水色)との内積しか非零になる要素はなく、結果としてAの水色と同じ行、Vの水色と同じ列のZの要素のみが非零となる)
![](img/attention.png)

The self reference structure can be achieved by this operation. $W_q$ is unit matrix
これによってattentionにおいて自己参照的な構造が得られることになります。Wqは単位行列でありKでその形をnxnに引き伸ばすようにすれば良さそうですがもともとのKey,Queryの意味付けと比べると消極的であるようにも見えます。別の言い方をするとattentionでは1回の繰り返しでより多くの自己参照をしていると言えます。
関数マップにおけるεの調整もモデル内部で行っていると言えるかもしれません。

The above example is only one element of each rows is non-zero, in case multiple elements are non-zero FD can be thought as converter of multi valued function of probablistic distribution function (probablistic process).In transformers renormalization for probablistic distribution function is calulated by softmax and division by root d.
In this paper we only treat same weight parameter for each layers compare to FV and attentions, recently this is called recurrent transformer and paid attention with researchers[].

One of the interesting property of FD is hierachical structure of points. Fixed points are on diagonal line called type I, type II fixed points is depends of  type III fixed points refer to ...and so on.  [].
![](img/hierchical.png)

This hierrachical structure is not merely analogy of the one of natural/programming languages but coreesponds to deduction or in-context learning process of transformers. As following figure, functional dynamics can generate self similar fractal shaped function by adding matrix operation as in attention mechanism. 
The compsition of attention ($f\cdot f$) and MLP as operators makes self recuesivee fractal shaped function easily. Fig .  shows ssteps  to make make two identical map inside the region of s map. This fact also implies self similar structure of language related to folding mechanism of FD.
![](img/FMAP_INCURSIVE.png)

The original form of FD only consists of function apply( $\cdot$ ),addition (+) and multiplication of constant value $\epsilon$ this restrict related to logic structure which transformers can calculete as following chapter.

### Category Theory
As described above,section FD and attention can be treated as some kind of Category and it should have ability to explain and evaluate functions. Lambda calculus treats functions as same as variables. All calculation in is multiple steps of evaluations(eval) and applications(apply) of formulars.
Eval is so called charactor string as a formular and calucule this, apply is the process that substitutiig eval's result to other formular. This eval-apply loop is common at the various field of computer programming.
Lambda calcuals has three rules, alpha conversion beta reduction and eta conversion. Alpha conversion is just replacement of bound variable names. Beta reduction is application of a function described by $(\lambda x. f) b=f(b)$ in usual notation. Eta conversion is desciribed as $(\lambda x. f) x=f $, here rhs and lhs are same function (constant). This is corresponds to extentionality definition of functions sets theorem.

Lambda calculus is formulate by using Cartesian closed category (CCC) which have product $X \times Y $ of tow objects X,Y and exponential object $X^Y$. There is natual bjiction $Hom(X \times Y,Z) \simeq Hom(X,Z^Y)$. There is one morphism called carring $\lambda g$ for all g and 
morphism $Z^Y→Z$ is evaluation of program(S-expression in LISP),this is coresspons to calculation of Attention matrix from Q and K, $f\cdot f$ of functional dynamics.

$$\begin{CD}
A @>{f}>> B \\
@VV{\lambda g}V @VV{g}V \\
C @ C @ .
\end{CD}$$

There is another least restricted category called Symmetric Monoid Closed Category(SMCC). SMCC do not have diagonal morphism. Intuidively diagonal morphism and its dual is copy and delete operation. When logic and proof process changes called linear logic.
This condition is common when the objects are vector space and morphisms are linear transformation because $X\times X=X^2$ is nonlinear. The category called $\bf{Vect}$.

Be aware with cardinality of exponential object is larger than the cardinality of objects. Lawvere's fixed-point theorem[] says .

Markov category(MC) is a modeling of probablistic calculation and statistical inference and induction. The object are probablistic distributions, the morphism are transition kernels between distributions. Generally MC is not CCC, 

Topos is defined CCC which has subobject classifiers $Sub: C^{op}\rightarrow Set$.

Intuitively function f and g of FD corresponds to morphism, the functor is functional dynamics.  In other formulation f,g are objects, functional dynamics itself is morphism and the functor is parametrize by $\epsilon$. By restricting the formular of FD linear interpolations as in the original paper[][], category theory can explation its parameters $\epsilon$. Functor between FD and parameter $\epsilon$ can be thought natual transformation.

Yoneda's lemma explains the relation between the behavior and parameters FD. According to Yoneda's lemma, for a category C, and a functor from C to othre category D.
$F(x) \leftrightarrow , x \in C, F: C^{op}$
C^{op}$ is the opposite category of C which morphics is reverse directino to C.

Hom functor $Hom_C(-,X)$ maps object A to a set of morphisms $Hom_C(A,X), morphism f: to set of morphism of morphism .
There are natural transformations $Nat(h_X,h_Y) (h_X \rightarrow h_Y)$. Yoneda's lemma states for any functor F $Nat(h_X,F) \simeq F(X) $.
This means a natrual transformation between FD F and G corresponds to a specific parameter. In case of linear interpolation FD, this is parameter $epsilon$. Here we identify attention matrix dynamics along layers is functional dynamics. The change of attention matrix dynamics along to layers can be said natural transformation and there is a specific object(values) maped by a functor F.

### Linear Logic, Linear lambda calculus
Linear logic is restriction of usual mathematical logic which only allows finite use of propositions during a deduction.

operator $A \multimap B$ means is linear implication, which signifies "deriving a conclusion by consuming a premise exactly once".

## Formulation
This chapter explains calculation ability of transformer based on the idea of functional dynmics and composition of specific cateories related to linear logic. 
As explained above, exponential object can be understood as "functions of functions" in function dynamics between 1-dimentional space i.e. infinite-dimentional vector space, a function represent as a graph on 2D space,especially one of those functions of function.
At the infinite limit of number of element of attention matrix, it becomes 2D space.

### Attention is not CCC, relations with Functional dynamics, attention matrix and  expornential object
Assume that input vector of transformer x, or each row of the products between weight matrix  $Q=W_Q, K=W_K$ represents probablistic 
distribution. Attention matrices are understood as markov transition kernel, this is markov category.
Transformers are represented as composition of SMCC and Markov category.
Vectors and Matrices muptiplied by weight matrix $W_k,$W_q,$W_V$ are reguearded as parameters of output vectors $Para(Vect)$.

We show CCC has diagonal morphism $A \rightarrow  A \times A$. Because this is not linear transformation, categoryt $\bf{Vect}$ is not CCC but 
of output has linear relation. Not to destroy linear structure at softmax. For counter expample of $\bf{Vect}$ is CCC incommutable

#### softmax and Markov category
 In attention mechanism 
 can be thought as probability disutribution of  and Markov category whose objects are probability distribution
 $softmas(x)/\sqrt{d}$ here we call this simply softmax. 

### Transformers as composition of SMCC and Markov category
is a composition of SMCC and Markov category
The output of this category is .
#### Typing the FV
In the reference implementation, an FV is computed as a sum over top heads of the out-projection of head activations at the last token, yielding an element of $\mathbb{R}^{d}=\mathbb{R}^{\mathrm{resid\_dim}}$. It is therefore literally a **point** of a parameter object $P_{FV}\subset\mathbb{R}^d$, *not* an element of the internal hom $X\multimap Y\cong\mathbb{R}^{d\times d}$. This dimension gap must be bridged by an explicit **realization map** $\Phi:P_{FV}\to(X\multimap Y)$.

The task index $t$ is **not** a parameter but *data*: it is inferred from the context, not a free weight. The weights $W_{q,k,v,O}$ are shared across all tasks and therefore cannot by themselves select $t$; only the context can. We thus separate two morphisms:

- **Extraction** $E:\mathrm{Ctx}\to P_{FV}$, reading the in-context examples once and producing the point $v_t=E(\mathrm{ctx}_t)$; the weights are the *static* parameters of $E$.
- **Application** $A:P_{FV}\otimes X\to Y$, defined via $\Phi$ and $\mathrm{ev}$.

Robustness of the FV to the choice of insertion site and head set is evidence that the conserved quantity is the point $v_t$ (a site-independent datum), supporting the *point* typing over a position-dependent section.

#### The linear approximation and multiplicative intuitionistic linear logic (MILL)
To reach linear $\lambda$-calculus we must keep the *bilinear* value path while treating the softmax pattern as frozen (a query-independent constant), because a fully linearized forward pass collapses the FV intervention to an affine translation and destroys $\multimap$. Under this approximation the ambient SMCC is $(\mathbf{Vect}_k,\otimes,I)$ with internal language the multiplicative intuitionistic fragment MILL $(\otimes,\multimap,I)$. The absence of a $\otimes$-diagonal is exactly the linearity discipline "each resource used once".

#### ICL as a single linear $\lambda$-term
With $E$ and $\Phi$ as constants, and using the currying isomorphism $\mathrm{Hom}(C\otimes A,B)\cong\mathrm{Hom}(C,A\multimap B)$:

$$
\dfrac{c:\mathrm{Ctx}\vdash \Phi(E\,c):X\multimap Y \qquad x:X\vdash x:X}
{c:\mathrm{Ctx},\,x:X\vdash (\Phi(E\,c))\,x:Y}\;(\multimap\text{-elim}=\mathrm{ev})
$$

$\boxed{\;\lambda c.\,\lambda x.\,(\Phi(E\,c))\,x\;:\;\mathrm{Ctx}\multimap(X\multimap Y)\;}$

Because $\multimap$-elimination merges disjoint contexts, both $c$ and $x$ are used exactly once, so the term is well-typed in linear $\lambda$-calculus. The reading is: the context is compiled to a *function-typed value* (the type $\mathrm{Ctx}\multimap(X\multimap Y)$), the FV $E(c)$ is a reified **function reference** (a first-class datum), $\Phi$ resolves the reference into a procedure, and $\mathrm{ev}$ applies it. This makes precise the FV literature's statement that the vector *triggers* rather than *performs* the task: $\text{FV}=E(c)$ is the point; $\text{procedure}=\Phi(E(c))\in X\multimap Y$.

An honest caveat: $\Phi$ is an additional, non-automatic structure; whether the downstream effect of an FV is genuinely a multiplicative modulation (eval) rather than a constant additive bias is an empirical question, addressed in §6.

### Hypothesis: the restriction of linear logic is partiallyt recorved by residual connections
Linear logic restricts using a proposition (or a fact) only once a during deduction process.
This makes the efficiency of deduction per one layer lower,  but makes mutch simpler deduction program as in human programming using spesicif language like Rust[].

### The total formulation
The main statement of this paper is drawn as transformers are mutial adjoint between piecewise-linear, Cartesian category corresponds to MLP and linear category corresponds to attention, expecially an attention matrix is a composition of Markov category and SMCC.
The data flow in an attention matrix as composition of Markov category and SMCC is depicted as fig.

$x \xrightarrow{W_q,W_k} (Q,K) \xrightarrow{softmax,carring} A \simeq Hom(X, \multimap Y) \xrightarrow{eval} C$

$ Kl(D)(pos,pos) \ni A \simeq (internal)Hom(V\multimap V)$

$A \otimes V \rightarrow C$

Here $KL(D)$ is Kleisli category and $D$ is distribution monad. $D$ and $Kl(D)$ is the category which have kernel as morphism.
The function of MLP has not shown here. Actual function is numerical experimentally decided.


### proofs of theorems formulation
The formal proofs of above statements written in lean is in appendix.
- Theorem 1 a layer of transformer is Kleisli morphism of composit monado M.
- Theorem 2 eval-apply is unit/counit of adjoint,  the type of $\lambda$c.$\lambda$x.(Φ(Ec))x is linear $\lambda$ term.
- Theorem 3 residual connection is written by (co)diagonal of biproduct, this is not !.
- Theorem 4 The two roles of the attention matrix, $Kl(D)(pos,pos)$ and $Hom(V\multimap V)$ connected by a functor.

## Numerical Experiments 
We explained attention structure, redisual connection and softmax in above forumulations  But MLP has not yet explained.
There is a statement that the function of MLP in transformer is key-value retrive [].Here we experimentally evaluate this hypothesis.
### The first experiment 
Here we show the result of relation between residual connection strength and ablity of reuse intermediate values. This hypothesis means the correlation between reuseage number $r$ and degration of model without redisual connection.

Each architectures trainined independely, the evaluations of same weight is done with continuous change of residual coefficient $alpha$.

#### Results

##### 実験1.1 :構成ごとに別訓練(分布シフトを排除)
残差を完全に除くと(none)、rに関わらずすべてchance近傍。残差ありは全rで完璧。これは「残差=深さ軸の加法的 ⊕-copy が値の運搬を担う」という予測と整合します。

一方「r が増えるほど劣化が大きい(r × 残差の交互作用)」は観測されませんでした ― none は r=1 から既に崩壊しており、r依存性がない。これは仮説の部分的な反証で、残差が担うのは「再利用回数に応じた fan-out」ではなく、そもそも値を深さ方向に運ぶこと自体らしい。この区別は理論にとって重要で、加法的 copy は「r 回の再利用を可能にする」のではなく「1回でも値を下流に届ける」ための配管だ、ということになる。

In this experiment r interaction effect is not observed. This means neither additive copy and Markov copy works as copy function solely.
##### 実験1.2: 同一重み・α 連続変形

full 残差で訓練したモデルの重みを一切変えず、x↦αx+f(x) の α だけ動かす(α=1 が訓練時):
```
α	q0	q1	q2
1.00	1.000	1.000	1.000
0.90	1.000	1.000	1.000
0.75	0.954	0.940	0.922
0.50	0.178	0.155	0.124
0.25	0.101	0.099	0.098
0.00	0.095	0.087	0.100
```

「同一重み・残差のみ異なる LLM の出力は異なりうるか」への答えは Yesでα=0.75 で後段クエリほど劣化が大きい(0.954→0.922)という微弱な勾配があり、深さ方向の運搬が徐々に効かなくなる描像と整合します。

### The second experiment 
Another question is that wehre outstanding performance of transformers comes from and is it related to category theoretical sturucture shonw in this paper or not. There is another possibility that architecture of network is not main cause of performance, specific learned weight values are essential for the performance. To solve this problem, we prepare an experiment comparing accuracy of retrival task with recurrent neural network (RNN) and state space model (SSM), which transformers good at and tatget taks of function vector study[].

#### Results
### 結果 2.1 ― 三アーキ比較(m=4、chance=0.167)

| Transformer | RNN(GRU) | 対角SSM |
|---|---|---|
| **1.000** | 0.405 | 0.322 |

**retrieval で Transformer が固定状態モデルに圧勝**。任意の過去の束縛を**アドレス指定**できる Transformer と、文脈を固定サイズ状態に圧縮してしまう RNN/SSM の差。これは turn 32 の逆 ― 状態追跡(Transformer が苦手)ではなく retrieval(Transformer が得意)を測ったので、比較優位が正しい向きに出ました。

### 結果 2.2 ― 構成要素 ablation(Transformer、m=4)

| intact | freeze_attn(動的 routing を殺す) | linear_mlp(eval 実現 Φ を殺す) |
|---|---|---|
| 1.000 | **0.284** | 0.855 |

**明確な分離です。** m=4 で Transformer=1.000、RNN=0.405、SSM=0.322(chance 0.167)— retrieval で Transformer が固定状態モデルに圧勝。次に ablation で「どの構成要素が担うか」を切り分けます。**二重乖離が出ました。** 結果をまとめます。Retrieval / 動的スパース性の検証実験を実装し、CPU で予備実行しました。設計の核心は turn 34 の通り「retrieval を解けること」でなく「**どの構成要素が担うか**」の分離です。

理論は「routing = softmax(Markov 核)、apply = eval」と予測しており、retrieval はちょうどその **routing 半分を行使**するタスクです。attention 凍結でそれが崩れたのは、理論が予測する構成が retrieval で**発揮されている**証拠 ― すなわち破れでなく構成の成功。「retrieval で勝つのは Markov(softmax=動的 routing)+ SMCC(eval)構成の成功であって破れではない」が裏づけられました。



## Discussion and Perspectives

### Related works

There is some attempt to prove transformers have same ability as Universal Turing Machine.
Th basis of these researtches, dyck language and its property is also important .Insted of use S-expression, using dyck language makes 
They define Attention and MLP parametrized mprphics or functor $\rm{Para(Vect)}$.
Then  define transformers and as free monad transformation [].
In there approach ,meta-prgramming feature of attention given by exponential object is not descripted.

Analysing the relation between lamada calculas and transformers are another dicection, 

Another formilization of transformers is based on topos. 
they first showed attention is exponential morphism of a category and has $lambda$ calculation ability as we shown. And using partical linear functions (PL) this property. This is natual condition because usual transformers and DNNs use ReLu as activation function.
But they ommits nonlinearity of the function such as softmax. Approximation of softmax function by PL is avairable but the probablistic meaaing of element of attention matrix is broken. The function of MLPs are not also explained well. 
Actually they apply pretopos insted of topos in the discussion. The assumption and application range is different from this paper but core idea, exponential object as meta-function is same.

Our statement result different perspective than []and []. Reguarding transformer as linear logic. Another difference of our papaer and above research is varification hypothesis by numeical and formal experiment using programming language such as python and lean. 

Along to FV, there are spesific change meaning or expression of words to inherent type.These head do not convert just words like words2vec, but function or morphism between they called function vector. FV can be understood as 

One-step calculation ability is also important question. If complex lambda formular(or S-expression) can be evaluate and applied at one time. Make one layer wider is more efficient than more layers. This is expecially important restrict speed, power and circuit footprint condition. In chapter[] we experimentally result both MLP and softmax has retrive function, the unbalance of attentions and MLPs is 

Multi Head attentions (MHA) are also key part of transformer performance. But it is not discussed and evaluated well in this paper. Pararlell architecture may achieved different values assignment to same expression and reduction. MLPs after concat work as selecting and merging the results, which could be decided this assumption is correct or not experimentally. Mixture-of-expert is same as MHA but larger structure.

Empirical experiments and direct explation of transformers' language processing ability corresponds to the architecutuer is important research region.
Several studies[][] are using S-expression or Dyck language to explain dettection ability of hierachical language and its limitation. As an extantion of function vector, detecting and processing meta functions described in usual programming language like S-expression is future work.

### Learnability
The success of transformers is not only higher order function programmability and in-context learning, but learnablity and avoiding local minimum, overfitting are also significant properties and affect to large application area industries.
For example, Edge of chaos hypothesis states highset learning speed is achieved when learning rate is on critical point[].
In other studies, attentions as a component of transformers tends to cluster in reccuerent structure. On the other hand MLP suffers from chaotic separation of phase spate[] which cause poor classification resulst.  As combinations of attention and MLP, transformers can be adjust learning dynamics properly to reach low loss function solution speedy. In this case changing the ration between attentions and MLPs and measure prediction performance of learned parameters is simple experiment to detect the function of edge of chaos flow[].

### Learning process in category theory
In this paper, we only show formulation and explanation of inference and generation of fuction Transformers and ability higher logical property.
To extend categoric theoretical view to the learnability of transformer, to explain this dynamical systems view required 
Because learning process cannot tread as natural transformation. Actually, 2-category which has morphism of morphism as an objcet, is nesesssary to explain fic

On the other hand ICL can ben treated as linear $\lambda$-calculas based on leanr category. Where the difference comes from is 

Using Cartesian reverse derivative category(CRDC) which has is an answer. Discriminaiton 2 kinds of Jacobian $R[f]$(partial derivativs of weight parameter) and $D_A[f]$(derivatives of layer input vector), are different variables but they are connected with chain rune of differnetation. Lyapunov spectrum ,eigen values of these Jacobians rule dynamics of neural networks.

Differencial structure, spectrum and topology are additional structure of CRDC and can be described by using vocablaries of basic category theory such as functor or natural transformation.

Whether the extra structure reduces to categories/functors/natural transformations, in the same sense that 2-categories are $\mathbf{Cat}$-enriched and $\infty$-categories are $\mathbf{sSet}$-enriched:

| Added structure | Basic-vocabulary formulation | Ground required externally | Reducibility |
|---|---|---|---|
| Differential | Tangent category: functor $T$ + natural transformations $(p,0,+,c,l)$ + limit preservation | (none essential) | Fully internal |
| Spectral | Dagger category + biproducts (contravariant $\dagger$, natural isos) | Scalar object $\Bbbk=\mathbb{C}$, algebraically closed and complete | Structure reducible; eigenvalue *content* is ground |
| Topological equivalence | $\mathbf{Top}$-enrichment + preservation of the $\mathbb{R}$-action (flow) | Base $\mathbf{Top}$ (or condensed) and a time object | Only via enrichment; not internalized |

So all three are expressible with categories, functors and natural transformations — but, exactly as for 2- and $\infty$-categories, spectral theory and topological equivalence require *choosing an enrichment base* ($\mathbb{C}$-linear complete dagger; $\mathbf{Top}$). Only the differential layer is purely internal (tangent categories). The minimal categorical setting for the bifurcation program is therefore: tangent category (differential, internal) + $\mathbb{C}$-linear complete dagger-biproduct enrichment (spectral, chosen ground) + $\mathbf{Top}$-enrichment or flow $\mathbb{R}$-action (topological, external). The most tractable route is to phrase grokking's saddle-to-saddle as a loss of invertibility / imaginary-axis crossing of the state Jacobian $D_A[f]$ in the dagger subcategory, which uses only the first two layers; topological conjugacy failure then follows via Hartman–Grobman.

In general, considering network architectures in points of view of dynamical systems and category theory is useful for the performance and its limit.
Especially CRDC relates to and describe lyapunov spectrum , bifurcation, learning dynamics like grokking is important question.

## Conclusions
In this paper, we show the correndence between lambda calculas and transformer, functional dynamics and explain the linear calculation ability in-context of transformers is composed Markov and SMCC categories.
### Reference
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Resnet](https://arxiv.org/abs/1512.03385)
- [Functional dynamics. I: Articulation process](https://cir.nii.ac.jp/crid/1360574095440074752)
- [Functional dynamics: II: Syntactic structure](https://www.sciencedirect.com/science/article/abs/pii/S0167278900002037)
- [Function Vectors in Large Language Models](https://functions.baulab.info)
- [Functional Attention](https://arxiv.org/abs/2605.31559)
- [Besic Category Theory](https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/leinster-book2.pdf)
- Markov cat https://www.sciencedirect.com/science/article/pii/S0001870820302656?via%3Dihub
- CRDC [Reverse derivative categories](https://arxiv.org/abs/1910.07065)
- [Introduction to Linear Logic](https://www.brics.dk/LS/96/6/BRICS-LS-96-6.pdf)
- [programs as singular](https://arxiv.org/abs/2504.08075)
- [Rust programming language](https://rust-lang.org/)
- [The Topos of Transformer Networks](https://arxiv.org/abs/2403.18415)
- Endofunctor[Endofunctor Self-Attention as a Parametric Endofunctor: A Categorical Framework for Transformer Architectures](https://arxiv.org/abs/2501.02931)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
https://learnmechinterp.com/topics/mlps-in-transformers/
- [Universal Transformers](http://arxiv.org/abs/1807.03819)
- [UTM critics](https://lifeiscomputation.com/transformers-are-not-turing-complete/)
- [Self-Attention Networks Can Process Bounded Hierarchical Languages](https://arxiv.org/abs/2105.11115)
- [Lawvere's fixed point theorem](https://ncatlab.org/nlab/show/Lawvere%27s+fixed+point+theorem)
- [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819)
## Aknowledgement
We thank very useful discussion with Dr. Kunihiko Kaneko and Dr. Kai Nakaishi.

## Appendix
### Appendix A formal proofs of theorems in lean
Common sector
```lean
import Mathlib.CategoryTheory.Monoidal.Closed.FunctorCategory.Basic
import Mathlib.CategoryTheory.Monoidal.Closed.Basic
import Mathlib.CategoryTheory.Monoidal.Braided.Basic
import Mathlib.CategoryTheory.Limits.Shapes.BinaryBiproducts
import Mathlib.CategoryTheory.Preadditive.Biproducts
import Mathlib.CategoryTheory.Limits.Shapes.IsTerminal
import Mathlib.CategoryTheory.Monad.Kleisli
import Mathlib.CategoryTheory.Monad.Basic
import Mathlib.CategoryTheory.Monad.Algebra
import Mathlib.CategoryTheory.Adjunction.Basic

open CategoryTheory CategoryTheory.MonoidalCategory CategoryTheory.Limits

noncomputable section
namespace TransformerCat

universe v u
variable {C : Type u} [Category.{v} C]
```
"noncomputable section" means following theorems do not caluculated specific values as functions.

- Theorem 1 The softmax routing is a Kleisli morphism of a monad D, and whole layer of transformer is Kleisli morphism of composit monado M = T ∘ D.
```lean
section KleisliLayer
variable (T D : Monad C)
variable {A B : C}

/-- In `Kleisli D`, a morphism A ⟶ B is by definition a base morphism
    A ⟶ D.obj B. The softmax/Markov routing `attn : A ⟶ D.obj B` (D = the
    distribution monad) is therefore literally a Kleisli morphism of D. -/
example (attn : A ⟶ (D : C ⥤ C).obj B) :
    @Quiver.Hom (Kleisli D) _ (A : Kleisli D) (B : Kleisli D) := attn

/-- Data representing a distributive law together with the composite monad
that its omitted Beck coherence equations are intended to induce. -/
/- The original blueprint used `True` in place of Beck's four coherence
axioms. Those placeholders cannot justify construction of a composite monad.
Until those equations are formalized, the honest interface must include the
resulting monad as data. -/
structure DistribLaw (T D : Monad C) where
  law : (D : C ⥤ C) ⋙ (T : C ⥤ C) ⟶ (T : C ⥤ C) ⋙ (D : C ⥤ C)
  composite : Monad C
  composite_toFunctor : (composite : C ⥤ C) = (T : C ⥤ C) ⋙ (D : C ⥤ C)

/-- The composite monad supplied by `DistribLaw`. -/
def composeMonad (l : DistribLaw T D) : Monad C := l.composite

/-- **The whole layer as a single Kleisli morphism of the composite monad.**
    Given the composite monad M = `composeMonad`, a layer
    `layer : A ⟶ M.obj B` is exactly a morphism `A ⟶ B` in `Kleisli M`.
    Thus the two categories (Markov `Kl(D)` and the value/SMCC part carried by T)
    are unified as morphisms of the single Kleisli category `Kleisli M`. -/
example (l : DistribLaw T D)
    (layer : A ⟶ ((composeMonad T D l) : C ⥤ C).obj B) :
    @Quiver.Hom (Kleisli (composeMonad T D l)) _
      (A : Kleisli (composeMonad T D l)) (B : Kleisli (composeMonad T D l)) :=
  layer

end KleisliLayer
```
- Theorem 2 eval-apply is unit/counit of adjoint,  the type of $\lambda$c.$\lambda$x.(Φ(Ec))x is linear $\lambda$ term.
```lean
section EvalApply
variable [MonoidalCategory C] [MonoidalClosed C]
variable {Ctx P X Y : C}

/- Extraction E : Ctx ⟶ P_FV and realization Φ : P_FV ⟶ (X ⊸ Y).
   Here `(ihom X).obj Y` is the internal hom X ⊸ Y. -/
variable (E : Ctx ⟶ P) (Φ : P ⟶ (ihom X).obj Y)

/-- The reified, realized function  f_t = Φ ∘ E : Ctx ⟶ (X ⊸ Y).
    This is the (linear) $\lambda$-abstraction / "choose" morphism. -/
def curriedFn : Ctx ⟶ (ihom X).obj Y := E ≫ Φ

/-- Application: uncurrying the reified function, X ⊗ Ctx ⟶ Y. -/
def applyMor : X ⊗ Ctx ⟶ Y := MonoidalClosed.uncurry (E ≫ Φ)

/-- **eval-apply.**  Applying the reified function equals "build it (X ◁ (E ≫ Φ)),
    then evaluate", where the evaluation `ihom.ev X` is exactly the counit of the
    adjunction `(tensorLeft X) ⊣ (ihom X)`.  This is the categorical content of
    `(Φ (E c)) x = eval (Φ (E c), x)`. -/
theorem apply_eq_build_then_ev :
    applyMor E Φ = (X ◁ (E ≫ Φ)) ≫ (ihom.ev X).app Y := by
  unfold applyMor
  rw [MonoidalClosed.uncurry_eq]

/-- **β-conversion** = the counit triangle: uncurry (curry g) = g. -/
theorem beta (g : X ⊗ Ctx ⟶ Y) :
    MonoidalClosed.uncurry (MonoidalClosed.curry g) = g :=
  MonoidalClosed.uncurry_curry g

/-- **η-conversion** = the unit triangle: curry (uncurry h) = h. -/
theorem eta (h : Ctx ⟶ (ihom X).obj Y) :
    MonoidalClosed.curry (MonoidalClosed.uncurry h) = h :=
  MonoidalClosed.curry_uncurry h

/-
  The term  $\lambda$c. $\lambda$x. (Φ (E c)) x  of linear $\lambda$-calculus, of type
  Ctx ⊸ (X ⊸ Y), DENOTES `curriedFn E Φ : Ctx ⟶ (ihom X).obj Y`, and its
  applied form denotes `applyMor E Φ`. Theorems `apply_eq_build_then_ev`, `beta`,
  `eta` are the semantic (SMCC) counterparts of the term's typing plus β/η.

  A *syntactic* soundness theorem — "this linear-$\lambda$ term is well-typed with each
  variable used exactly once, and its denotation is `applyMor`" — requires a
  formalized linear type system (contexts as multisets, ⊸-intro/elim, a
  no-contraction/no-weakening discipline) that Mathlib does NOT provide. That is
  a separate development; here we formalize the denotation only.
-/

/-- Naturality bookkeeping: uncurrying commutes with precomposition by E,
    i.e. the "choose then apply" pipeline composes as expected. -/
theorem apply_factor :
    applyMor E Φ = (X ◁ E) ≫ MonoidalClosed.uncurry Φ := by
  unfold applyMor
  rw [MonoidalClosed.uncurry_natural_left]

end EvalApply
```
- Theorem 3 residual connection is written by (co)diagonal of biproduct, this is not !.
```lean
section Residual
variable [Preadditive C] [HasBinaryBiproducts C]
variable {A : C}

/-- The additive diagonal Δ_⊕ : A ⟶ A ⊞ A (fan-out along the depth axis). -/
def diagAdd (A : C) : A ⟶ A ⊞ A := biprod.lift (𝟙 A) (𝟙 A)

/-- The additive codiagonal ∇_⊕ : A ⊞ A ⟶ A (write-back). -/
def codiagAdd (A : C) : A ⊞ A ⟶ A := biprod.desc (𝟙 A) (𝟙 A)

/-- **Residual as additive copy (clean form).**
    `biprod.lift (𝟙) f ≫ biprod.desc (𝟙) (𝟙) = 𝟙 + f`.
    The two branches Δ_⊕ produces are summed back by ∇_⊕ into a SINGLE
    resource `𝟙 + f`; this is why the residual copies additively but supplies
    no independent second consumption. -/
theorem residual_eq (f : A ⟶ A) :
    biprod.lift (𝟙 A) f ≫ biprod.desc (𝟙 A) (𝟙 A) = 𝟙 A + f := by
  simp [biprod.lift_desc]

/-
**Residual as Δ_⊕ ≫ (id ⊞ f) ≫ ∇_⊕.**
    Same statement, written through the diagonal / map / codiagonal, matching
    the string-diagram reading.
-/
theorem residual_eq_diag (f : A ⟶ A) :
    diagAdd A ≫ biprod.map (𝟙 A) f ≫ codiagAdd A = 𝟙 A + f := by
  simp +decide [ diagAdd, codiagAdd, ← Category.assoc ];
  grind +suggestions

/-
The original proposed theorem `no_tensor_diagonal_of_noncartesian` was
incorrect: in a preadditive monoidal category the family of zero maps is always
such a natural diagonal.  A counit, including its normalization and naturality,
is needed to derive the advertised obstruction.

A natural, normalized family of discarding maps would make the tensor unit
terminal. Hence such a family cannot exist when the tensor unit is not
terminal. This is the part of the obstruction to a uniform copying/discarding
comonoid structure that follows directly from non-cartesianness.
-/
omit [Preadditive C] [HasBinaryBiproducts C] in
theorem no_natural_discard_of_nonterminal_unit
    [MonoidalCategory C]
    (hNonterminal : IsEmpty (Limits.IsTerminal (𝟙_ C))) :
    ¬ ∃ ε : (A : C) → (A ⟶ 𝟙_ C),
        ε (𝟙_ C) = 𝟙 (𝟙_ C) ∧
        (∀ {A B : C} (g : A ⟶ B), g ≫ ε B = ε A) := by
  rintro ⟨ε, hunit, natural⟩
  apply hNonterminal.false
  refine Limits.IsTerminal.ofUniqueHom ε ?_
  intro X m
  simpa [hunit] using natural m

end Residual

```
- Theorem 4  The two roles of the attention matrix, $Kl(D)(pos,pos)$ and $Hom(V\multimap V)$ connected by a functor.
```lean
section RepresentationFunctor
variable (D : Monad C)

/-- **The representation functor `F_V` (Type-valued), FULLY PROVED functorial.**
    A probability kernel `A` is sent to the value-mixing operator on `pos ⟶ V`.
    `map_id` uses the unit of the monad and of the algebra; `map_comp` uses the
    multiplication, its naturality, and the algebra's associativity. No strength,
    no `sorry`. -/
def valuePresheaf (Valg : D.Algebra) : (Kleisli D)ᵒᵖ ⥤ Type v where
  obj X := X.unop ⟶ Valg.A
  map {X Y} A := fun val => A.unop ≫ (D : C ⥤ C).map val ≫ Valg.a
  map_id X := by
    funext val
    simp only [unop_id]
    -- Kleisli identity is the monad unit η; then η-naturality + algebra unit.
    show D.η.app X.unop ≫ (D : C ⥤ C).map val ≫ Valg.a = val
    rw [← Category.assoc, ← D.η.naturality val, Category.assoc, Valg.unit,
        Category.comp_id]
  map_comp {X Y Z} A B := by
    funext val
    -- opposite comp unops to reversed Kleisli comp
    --   (A ≫ B).unop = B.unop ≫_Kl A.unop = B.unop ≫ D.map A.unop ≫ μ ;
    -- expand D.map of the composite on the right, then μ-naturality + algebra assoc.
    show (B.unop ≫ (D : C ⥤ C).map A.unop ≫ D.μ.app X.unop)
            ≫ (D : C ⥤ C).map val ≫ Valg.a
        = B.unop ≫ (D : C ⥤ C).map (A.unop ≫ (D : C ⥤ C).map val ≫ Valg.a) ≫ Valg.a
    rw [Functor.map_comp, Functor.map_comp]
    simp only [Category.assoc]
    rw [D.μ.naturality_assoc, Valg.assoc]

/-- **Functoriality made explicit: the kernel action respects Kleisli identity.**
    `A = η` (the deterministic "stay put" kernel) acts as the identity operator. -/
theorem valuePresheaf_map_id (Valg : D.Algebra) (X : (Kleisli D)ᵒᵖ) :
    (valuePresheaf D Valg).map (𝟙 X) = id :=
  (valuePresheaf D Valg).map_id X

/-- **and respects Kleisli composition (Chapman–Kolmogorov ↦ operator comp).** -/
theorem valuePresheaf_map_comp (Valg : D.Algebra) {X Y Z : (Kleisli D)ᵒᵖ}
    (A : X ⟶ Y) (B : Y ⟶ Z) :
    (valuePresheaf D Valg).map (A ≫ B)
      = (valuePresheaf D Valg).map A ≫ (valuePresheaf D Valg).map B :=
  (valuePresheaf D Valg).map_comp A B

end RepresentationFunctor

```
### Appendix B: The relation between distribution moand D, Kleisli categgory $\mathrm{Kl}(D)$ and D-algebra category $\mathrm{EM}(D)$
#### 1. The Distribution Monad $D$

Let the base category be $\mathbf{Set}$, and define the finite-support version (the continuous version is discussed later):
$$D(X)=\Big\{\,p:X\to[0,1]\ \Big|\ \mathrm{supp}(p)\text{ finite},\ \textstyle\sum_{x}p(x)=1\,\Big\}$$

This is the set of finitely-supported probability distributions on $X$. The three-part package that makes it a monad:

- **Unit** $\eta_X:X\to D(X)$, $x\mapsto \delta_x$ (the Dirac measure, point mass). "Regard a definite value as a distribution."
- **Multiplication** $\mu_X:D(D(X))\to D(X)$, collapsing a distribution of distributions by averaging: $\mu(P)(x)=\sum_{q}P(q)\,q(x)$. This is exactly the **law of total probability**.
- **Functorial action** $D(f):D(X)\to D(Y)$ (for $f:X\to Y$) is the **pushforward**: $D(f)(p)(y)=\sum_{x:f(x)=y}p(x)$.

The monad laws (left/right unit laws and associativity) coincide with the **basic identities of probability**: the marginalization of Dirac measures and the associativity of mixing. Here $\eta$ is "making definite" and $\mu$ is "flattening a mixture."

**On the base category**: $D$ is **commutative** (the sampling order of two independent distributions does not change the result = Fubini) and **affine** ($D(1)\cong 1$; there is exactly one distribution on a one-point set). These two properties are what later make $\mathrm{Kl}(D)$ a Markov category.

**Variants**: the **Giry monad** $G(X)=\{$probability measures on $X\}$ over measurable spaces $\mathbf{Meas}$ (the continuous version — this is the right one for the real-valued logits of attention); the subdistribution monad for subprobabilities ($\sum\le 1$); and so on. Since softmax has finite support, $D$ suffices.

#### 2. The Kleisli Category $\mathrm{Kl}(D)$ — the Category of Stochastic Kernels

$\mathrm{Kl}(D)$ is the category that puts "stochastic morphisms" center stage.
- **Objects**: the same as the base category (sets).
- **Morphisms** $X\to Y$: functions $X\to D(Y)$, i.e. assignments of a distribution on $Y$ to each $x$ — **Markov kernels (stochastic kernels)**. For a finite set, a **stochastic matrix** $k(x)(y)=P(y\mid x)\ge0$ whose columns (or rows) sum to 1.
- **Identity morphism**: $\eta_X$, $x\mapsto\delta_x$ (deterministically pass through unchanged).
- **Composition** (Kleisli composition = **Chapman–Kolmogorov**):
$$(k'\circ k)(x)(z)=\sum_{y}k(x)(y)\,k'(y)(z)$$
the product of stochastic matrices. "Marginalize over the intermediate $y$ the transition from $x$ to $y$ and from $y$ to $z$."
- **Symmetric monoidal structure** (from the commutativity of $D$): $\otimes=$ the Cartesian product of sets, and the tensor of kernels = the independent joint distribution.
- **Markov category structure**: copy $X\to X\otimes X$, $x\mapsto\delta_{(x,x)}$ (deterministic duplication) and delete $X\to 1$. Because $D$ is affine, delete is unique (semicartesian); because it is commutative, it is symmetric monoidal. **This copy is non-natural** — for a genuinely stochastic kernel $k$, "copy then $k$" (a perfectly correlated pair $(y,y)$) does not agree with "$k$ then copy" (independent resampling). It is a copy that generates correlation.

Restricting $\mathrm{Kl}(D)$ to finite sets gives **FinStoch** (the category of stochastic matrices). **softmax$(QK^\top)$ is precisely a morphism of this category** — a stochastic kernel from query positions to key positions, a stochastic matrix $A$.

#### 3. $D$-Algebras (the Eilenberg–Moore Category $\mathrm{EM}(D)$)

Whereas Kleisli made "morphisms" the protagonist, $D$-algebras make "**a structure that consumes a distribution and returns a value**" the protagonist.

A **$D$-algebra** is a pair $(X,\alpha)$ with $\alpha:D(X)\to X$ satisfying two coherence laws:

$$\alpha\circ\eta_X=\mathrm{id}_X\qquad(\text{the point mass }\delta_x\text{ evaluates to }x)$$
$$\alpha\circ\mu_X=\alpha\circ D(\alpha)\qquad(\text{a distribution of distributions: flatten then evaluate = evaluate each then evaluate})$$

**Meaning**: $\alpha$ maps "a distribution on $X$ to a single element," i.e. it performs **taking a barycenter / expectation / convex combination**. In other words, a $D$-algebra = a set on which convex combinations can be taken.

**What concretely is a $D$-algebra**: the Eilenberg–Moore category of the finite-distribution monad is equivalent to the category of **convex spaces (abstract convex spaces)** (Fritz–Perrone et al.). The decisive example for us — **any real vector space $V$ is a $D$-algebra**, with structure map

$$\alpha_V:D(V)\to V,\qquad \alpha_V(p)=\sum_{v}p(v)\,v=\mathbb E_{p}[v]\quad(\textbf{expectation}).$$

More generally, a convex subset of a vector space is a $D$-algebra. The morphisms of algebras are affine maps (commuting with convex combinations = commuting with $\alpha$).

**Value mixing is exactly this action**: Because after softmax produces $A(x)\in D(\text{positions})$, the value mixing $A\!\cdot\!V=\sum_y A(x)(y)\,v_y$ is the $D$-algebra structure $\alpha_V:D(V)\to V$ of the value space $V$ applied to the distribution = the expectation.

$$\text{attention head}=\underbrace{\big[\mathrm{Ctx}\xrightarrow{\ \text{softmax}\ }D(\text{pos})\big]}_{\text{morphism of }\mathrm{Kl}(D)}\ \text{composed with}\ \underbrace{\big[D(V)\xrightarrow{\ \alpha_V=\mathbb E\ }V\big]}_{\text{action of a }D\text{-algebra}}.$$

Consistency with Kolmogorov theory: a random variable = a measurable function on the sample space $\Omega\to\mathbb R$, and the expectation $\mathbb E$ is the $D$-algebra (Giry-algebra) structure of $\mathbb R$ (or $V$). **softmax routes and expectation mixes** fits into the single phrase "kernel ∘ algebra action."

#### 4. The Relationship Between $\mathrm{Kl}(D)$ and $\mathrm{EM}(D)$ — Adjunctions and the Comparison Functor

The two are not unrelated; they are two resolutions of the same monad $D$. Every monad arises from an adjunction $F\dashv U$, and $D$ has two canonical ones.

- **Kleisli resolution**: $F_K:\mathbf{Set}\to\mathrm{Kl}(D)$ (free) $\dashv U_K$. $\mathrm{Kl}(D)$ is the category of **free $D$-algebras** only.
- **Eilenberg–Moore resolution**: $F_{EM}:\mathbf{Set}\to\mathrm{EM}(D)$, $X\mapsto(DX,\mu_X)$ (free algebra) $\dashv U_{EM}$ (forgetful, $(X,\alpha)\mapsto X$). $U_{EM}F_{EM}=D$ recovers the monad.

The **counit of $F_{EM}\dashv U_{EM}$ is exactly the algebra structure map** — $\varepsilon_{(X,\alpha)}:(DX,\mu)\to(X,\alpha)$ is $\alpha$ itself. So calling the expectation $\mathbb E$ "counit-like" in the previous group of turns was accurate: **value mixing = a component of the counit of the EM adjunction**.

And the **comparison functor** $K:\mathrm{Kl}(D)\to\mathrm{EM}(D)$, $X\mapsto(DX,\mu_X)$, is fully faithful, embedding $\mathrm{Kl}(D)$ as the **full subcategory of free algebras**:

$$\mathrm{Kl}(D)\ \hookrightarrow\ \mathrm{EM}(D)\qquad(\text{free algebras}).$$

So the two are the "free end ($\mathrm{Kl}$, kernels)" and the "full-algebra end ($\mathrm{EM}$, convex spaces)": the value space $V$ lives in $\mathrm{EM}(D)$ as a **non-free** $D$-algebra, and the attention computation feeds the kernel $A$ of $\mathrm{Kl}(D)$ into the structure map $\alpha_V$ of $V$, an object of $\mathrm{EM}(D)$ — a move that **straddles the two categories**. This is the $D$-side content of "a Transformer is not a single SMCC but a composite of a Markov category and an SMCC."

# 5. Summary Table, and Implications for Unification

| | $\mathrm{Kl}(D)$ | $\mathrm{EM}(D)=\mathrm{Alg}(D)$ |
|---|---|---|
| protagonist (morphism/object) | stochastic kernel (morphism) $X\to D(Y)$ | $D$-algebra (object) $(X,\alpha)$ |
| concretely | stochastic matrix (FinStoch) | convex space / vector space (with expectation) |
| Transformer | **softmax** = kernel $A$ | **value space** $V$, $\alpha_V=\mathbb E$ = value mixing |
| relation to the monad | subcategory of free algebras | full algebras, $U_{EM}F_{EM}=D$ |
| copy | non-natural (generates correlation) | convex structure preserved by affine maps |

The unified category (composite monad $M=T\circ D$)" also becomes visible here. To bundle the linear part $T$ and the distribution $D$ into a single monad, one needs a distributive law $DT\Rightarrow TD$, and its natural candidate is precisely **"expectation commutes with linear maps"** — the very fact that $\alpha_V$ is affine/linear. That the $D$-algebra structure is compatible with the linear structure of $V$ ($\mathbb E$ is linear) is the seed that generates the distributive law between $D$ and $T$. So pinning down the relationship between $\mathrm{Kl}(D)$ and $\mathrm{EM}(D)$ leads directly into the construction of the unified category (the contents of the previous turn's Lean `composeMonad`).

Writing down the distributive law $DT\Rightarrow TD$ from this linearity of expectation = the $D$-algebra is the next move toward unification and Lean formalization. If you want to proceed by concretely defining the candidate for that natural transformation (corresponding to "pushforward of expectation under a linear map," $D(Tf)\to T(Df)$), you can write out each component of the distributive law by taking $V$ as the free vector space monad $T$ and $D$ as the distribution monad.

