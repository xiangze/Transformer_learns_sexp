# Transformers as Functional Dynamics

## Abstract
We 

----

## Introduction

### Contributions of this paper
- point out equivalence between self-attention and functional dynamics, property category .
- explains the relation between self-attention and symmetric monoidal closed category (SMCC) $(\mathbf{Vect},\otimes,\multimap)$
eval-apply loop which is required for in-context learning.
- and numerical experiment 
MLP is 

## Preriminalies
### Transformers as self attention
layered
is 
context learning alibity and scalability of .
is large language models (LLM).
made various applications and theoretical explanations.

Residual connection (Resnet) is often used in LLM. The benefits of Resnet are not only preserving information of earlier layers during training and inference, but simplify loss landscape
near identical 

Function vector(FV) [] a concept embedded in LLM as a head of transformer. FV is portable among but layer of LLM.

### Functional Dynamics

Functional dynamics (FD)[] is introduced by
1 \-dimentional graph of a function.
$f'(x)=$
Generally, the dynamics of functions 
is 
hieralchy of fixed points and complex
By restricting the formular of FD linear interpolation as in the original paper, category theory can exlation 
its parameters $$.
### Category Theory


### Linear lambda calculus
lambda calculus is basis of functions as variables. 
All calculation in is multiple steps of evaluations(eval) and applications(apply) of formulars.
eval is 
three rules, alpha conversion , beta reduction and eta conversion.
this eval-apply loop is common at the various field of computer programming.
lambda calculus is formulate by using Cartesian closed category (CCC) which have product $X \times Y $ of tow objects X,Y and exponential $X^Y$.
There is natual bjiction $Hom(X \times Y,Z) \simeq Hom(X,Z^Y)$. There is one morphism called $\lambda g$ for all g

Linear logic is restriction of usual mathematical logic.

## Formulation
This chapter explains calculation ability of transformer based on the idea of functional dynmics and spesific cateory composition correnspnds to 
relation of linear logic.

## Experiment 

along to FV.

## Discussion
### Relateed works

## Conclutions and Perspectives
In this paper, we show the correndence between lambda calculas and transformer, functional dynamics and explain the linear calculation ability in-context of transformers is composed Markov and SMCC categories.
### Reference


- Deep Residual Learning for Image Recognition
- Visualizing the Loss Landscape of Neural Nets
