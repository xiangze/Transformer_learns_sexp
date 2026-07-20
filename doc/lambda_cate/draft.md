\# Transformers as Functional Dynamics, equivalency between lambda calculas and linear logic
Categoriy theoretical view of transformas and functional dynamics, threre ability of higher order calculation
## Abstract
In this papaer we show the equivalence of functional dynamics and self-attention mechanism, then linear interpolation type functional dynamics whose parameter is $\epsilon$ is equivalent to the functor between functinoal dynamics from Yoneda's lemma.
From the point of view the equivalence between attention and functional dynamics, 
In-context learning, higher order logic function ability which seems transfromers, especially the part of self-attention have can be explained by symmetric monoidal closed category (SMCC) which has exponenatioal object.
SMCC is not equivalent to Cartesian closed category (CCC), both has higher order $lamda$-calculas function but we explain SMCC obeys linear logic(linear $lambda$ calculation) which one property or value can be used only once during calculation or proof.

And we state residual connection, another component of transformers "copy" data from previous layer, the constraint of linear logic is weakend and comutation ability recovers from the original SMCC.

Moreover we show MLP(Multi layer perceptron) has a function to retrive information from key-value database experimentally.

Because softmax is applied attention matrix, we can define Markov category as the object is probablistic distribution function,
the morphism is converison like Markov kernel.

As a result, one layer of a Transformer composition of Markov category and SMCC caluculate linear lambda calculation, and the property can extend nonlinear conversion such as softmax.

Related numerical experiments and theorem formulation automatic prooves are provides.

Keywords:
category theory, Transformers, sefl-attention, function vectors, function dynamical, dynamical systems, linear λ-calculus, Markov categories

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

In this paper almost explaing about the relation between Transformers, lambda calculation and category theory. But the original idea and motivation about higher order fuction is Vector space and moprphism or category of metafunction(funcsions between funcions) is from dynamical system of functions, this is deeply related to learnability of transformers.

some theorems are written in Lean.

### Contributions of this paper
- Point out equivalence between self-attention and functional dynamics, property category .
- Explains the relation between self-attention and symmetric monoidal closed category (SMCC) $(\mathbf{Vect},\otimes,\multimap)$ eval-apply loop which is required for in-context learning and correspondence between markov category and attention matrix.
- Numerical experiment about MLP function, only Key-value retirival or not.
- proofs of formalizations witten in lean
## Preriminalies
### Attention mechanism, Transformers and their Components
Transformers are consists of several components, attention, MLP(multi layer perceptron ,FFN), softmax, residual connecctions and layer normalizations.
Attentions are product of  and input vector x.
MLP is composition of all-to-all vector product using matrix product and activation function such as Relu or softmax.
softmax function is usually used tu make attention matrix in conrast of Relu in the tail of MLP.
Residual connection (Resnet) is often used in LLM. The benefits of Resnet are not only preserving information of earlier layers during training and inference, but simplify loss landscape. Resnet with nearly identical matrix convertion are similar to differential operations  which is called neural ordinal differencial equiation(neural ODE).
Layer normalizations are another important part of transformers to regulize internal data.

Positional encoders are also important for identify the order of tokens which is encoded and put in attention mechanism.
Layered transformers is usually called large language models (LLM). LLMs have in-context learning ability[] and scalability of learning. LLMs and their variants made various applications and theoretical explanations.

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

 One of the interesting property of FD is hierachical structure of points. Fixed points are on diagonal line called type I, type II fixed points is depends of  type III fixed points refer to ...and so on.  [].
This hierrachical structure is not merely analogy of the one of natural/programming languages but coreesponds to deduction or in-context learning process of transformers. As following figure, functional dynamics can generate self similar fractal shaped function by adding matrix operation as in attention mechanism. 
The compsition of attention ($f\times f$) and MLP as operators makes self recuesivee fractal shaped function easily. Fig .  shows ssteps  to make make two identical map inside the region of s map. This fact also implies self similar structure of language related to folding mechanism of FD.

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

Be aware with cardinality of exponential object is larger than the cardinality of objects "Lawvere's fixed-point theorem"[].

Markov category(MC) is a modeling of probablistic calculation and statistical inference and induction. The object are probablistic distributions, the morphism are transition kernels between distributions. Generally MC is not CCC, 

### Linear Logic, Linear lambda calculus
Linear logic is restriction of usual mathematical logic which only allows finite use of propositions during a deduction.
Topos is CCC which has .

operator $A \multimap B$ means is linear implication, which signifies "deriving a conclusion by consuming a premise exactly once".

## Formulation
This chapter explains calculation ability of transformer based on the idea of functional dynmics and spesific cateories composition related to  relation of linear logic.
As explained above, exponential object can be understood as "functions of functions" in 
in 1 dimentional space, a function represent as a graph on 2D space,especially  one of those functions of function 
The problem is 

Assume that nput vector of transformer x, or each row of the products between weight matrix  $Q=W_Q, K=W_K$ represents probablistic distribution. Attention matrices are understood as markov transition kernel, this is markov category.
Transformers are represented as composition of SMCC and Markov category.

### functional dynamics, attention matrix, expornential object

### Attention is not CCC
We show CCC has diagonal morphism $A \rightarrow  A \times A$. Because this is not linear transformation, categoryt $\bf{Vect}$ is not CCC but 
but of output has linear relation. Not to destroy linear structure at softmax.

#### softmax and Markov category
 In attention mechanism 
 here we call this simply softmax. 
 can be thought as probability disutribution of  and Markov category

### Transformers as composition of SMCC and Markov category
composition of SMCC and Markov category
The output of this category

## The restriction of linear logic and its recorvery by residual connections
Linear logic restricts using a proposition (or a fact) only once time during deduction process.
This makes the efficiency of deduction per one layer lower,  but makes mutch simpler deduction program as in human programming using spesicif language like Rust[].

## The total formular
The main statement of this paper is drawn as attention matrix is 
the data flow in a layer of transformer as composition of Markov category and SMCC is depicted as fig.
$KL(D) \rightarrow (Vect, ) \rightarrow (Vect, ) \rightarrow (Vect, ) $

$x \xrightarrow{W_q,W_k} (Q,K) \xrightarrow{softmax,carring} A \simeq Hom(X, \multimap Y) \xrightarrow{eval} C$

$ Kl(D)(pos,pos) \ni A \simeq (internal)Hom(V\multimap V)$

$A \otimes V \rightarrow C$

Here $KL(D)$ is Kleisli category and $D$ is distribution monad. $D$ and $Kl(D)$ is the category which have kernel 
The function of MLP has not shown here. Actual function is numerical experimentally decided.

## Experiment 
We explained attention structure, redisual connection and softmax in above forumulations  But MLP has not yet explained.
There is a statement that the function of MLP in transformer is key-value retrive [].Here we experimentally evaluate this hypothesis.

Here we show the result of relation between residual connection strength and ablity of reuse intermediate values. This hypothesis means the correlation between reuseage number $r$ and degration of model without redisual connection.

In this experiment r interaction effect is not observed. This means neither additive copy and Markov copy works as copy function solely.

### proof of theorems formulation
- Theorem 1 a layer of transformer is Kleisli morphism of composit monado M.

- Theorem 2 eval-apply is unit/counit of adjoint,  the type of λc.λx.(Φ(Ec))x is linear $\lambda$ term.

- Theorem 3 residual connection is written by (co)diagonal of biproduct, this is not !.

The last statement Theorem 3 can not formalize by lean (uses sorry tactics).

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

### Learnability
The success of transformers is not only higher order function programmability and in-context learning, but learnablity and avoiding local minimum, overfitting are also significant properties and affect to large application area industries.
For example, Edge of chaos hypothesis states highset learning speed is achieved when learning rate is on critical point[].
In other studies, attentions as a component of transformers tends to cluster in reccuerent structure. On the other hand MLP suffers from chaotic separation of phase spate[] which cause poor classification resulst.  As combinations of attention and MLP, transformers can be adjust learning dynamics properly to reach low loss function solution speedy. In this case changing the ration between attentions and MLPs and measure prediction performance of learned parameters is simple experiment to detect the function of edge of chaos flow.

### Learning process as category
In this paper, we show explanation of inference and generation of fuction Transformers and ability higher logical property.
To extend categoric theoretical view to the learnability of transformer, to explain this dynamical systems view required 
Because learning process cannot tread as natural transformation. sdo 2-category which has morphism of morphism as an objcet, is nesesssary to explain  
Using Cartesian reverse derivative category(CRDC) which has is an asswer. Discriminaiton 2 kinds of Jacobian $R[f]$(partial derivativs of weight parameter) and $D_A[f]$(derivatives of layer input vector), are different variables but they are connected with chain rune of differnetation. Lyapunov spectrum ,eigen values of these Jacobians rule dynamics of neural networks.

Differencial structure, spectrum and topology are additional structure of CRDC and can be described by using vocablaries of basic category theory such as functor or natural transformation.

How CRDC relates and describe lyapunov spectrum , bifurcation, learning dynamics like grokking is important question.

## Conclusions
In this paper, we show the correndence between lambda calculas and transformer, functional dynamics and explain the linear calculation ability in-context of transformers is composed Markov and SMCC categories.
### Reference
- Attention is all you need(https://arxiv.org/abs/1706.03762)
- Resnet https://arxiv.org/abs/1512.03385
- [Functional dynamics. I: Articulation process](https://cir.nii.ac.jp/crid/1360574095440074752)
- [Functional dynamics: II: Syntactic structure](https://www.sciencedirect.com/science/article/abs/pii/S0167278900002037)
- FV Function Vectors in Large Language Models(https://functions.baulab.info)
- Functional Attention https://arxiv.org/abs/2605.31559
- [Besic Category Theory]https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/leinster-book2.pdf
- Markov cat https://www.sciencedirect.com/science/article/pii/S0001870820302656?via%3Dihub
- CRDC [Reverse derivative categories](https://arxiv.org/abs/1910.07065)
- LL [Introduction to Linear Logic](https://www.brics.dk/LS/96/6/BRICS-LS-96-6.pdf)
- [programs as singular]https://arxiv.org/abs/2504.08075
- [Rust programming language]https://rust-lang.org/
- Topos[The Topos of Transformer Networks]https://arxiv.org/abs/2403.18415
- Endofunctor[Endofunctor Self-Attention as a Parametric Endofunctor: A Categorical Framework for Transformer Architectures] https://arxiv.org/abs/2501.02931
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
https://learnmechinterp.com/topics/mlps-in-transformers/
- UTM []
- Dyck
- [Lawvere's fixed point theorem](https://ncatlab.org/nlab/show/Lawvere%27s+fixed+point+theorem)