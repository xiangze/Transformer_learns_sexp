# The Eval–Apply Structure of Transformers: A Categorical Account of Function Vectors, Linear λ-Calculus, and the Learning–Dynamics Divide

**Abstract.** We develop a categorical framework for the hypothesis that multi-layer Transformers realize an *eval–apply loop*: attention performs evaluation and MLP performs realization/application, with in-context learning (ICL) producing reusable function representations. We argue that the *ambient category* of a single Transformer layer is not one symmetric monoidal closed category (SMCC) but a composite of three structures with distinct copy behaviour: the Kleisli category of the distribution monad $\mathrm{Kl}(D)$ (softmax, as a Markov category), the SMCC $(\mathbf{Vect},\otimes,\multimap)$ (value mixing and evaluation), and a non-linear realization map together with the additive biproduct structure $(\mathbf{Vect},\oplus)$ (MLP and residual stream). Within the linear approximation, we type a Function Vector (FV) as a *point* of a parameter object $P_{FV}\subset\mathbb{R}^d$, give a realization map $\Phi$ that curries it into the internal hom, and show that one forward pass of ICL is well-typed as a single **linear** λ-term $\lambda c.\lambda x.(\Phi(E\,c))\,x:\mathrm{Ctx}\multimap(X\multimap Y)$ in multiplicative intuitionistic linear logic (MILL). We distinguish three notions of copy — the forbidden multiplicative diagonal, the additive residual diagonal, and the non-natural Markov copy — and argue that none of them supplies the exponential modality $!$, so the model has no capacity for unbounded reuse. Small controlled experiments locate the cross-position (copy-like) interaction in softmax and the applied value in the final MLP, consistent with *eval = attention, apply = MLP realization*. We then show that *learning is not a natural transformation* between Transformer functors — a fact independent of architecture — but a 2-cell (reparametrisation) in the 2-category $\mathrm{Para}(\mathbf{Vect})$, whose differential structure requires a Cartesian reverse derivative category (CRDC). This separates two Jacobians — the parameter Jacobian $R[f]$ that drives learning and the state Jacobian $D_A[f]$ whose spectrum governs the network's dynamics — clarifying why learning dynamics (e.g. grokking) and Lyapunov spectra are *related but distinct*. Finally we position dynamical bifurcation inside CRDC, showing which added structures (differential, spectral, topological) reduce to the basic vocabulary of categories, functors and natural transformations, and survey the implementations that instantiate parts of this program.

*Keywords:* category theory, Transformers, function vectors, linear λ-calculus, Markov categories, reverse derivative categories, tangent categories, mechanistic interpretability.

---

## 1. Introduction

A growing body of work applies category theory to neural architectures, ranging from architecture-generating frameworks to mechanistic accounts of specific behaviours. This paper concerns a specific hypothesis about *behaviour*: that a multi-layer Transformer implements an **eval–apply loop**, in which attention performs evaluation (assembling a function with its argument) and the MLP performs application/realization (turning a representation into an applied value), and in which in-context learning (ICL) yields compact, reusable representations of tasks.

The most direct empirical anchor is the work on **Function Vectors** (FVs) [Todd et al. 2024], which shows that autoregressive Transformers develop a rudimentary form of *function reference* — a single vector that triggers execution of a task without itself performing it — and which explicitly invokes Church's λ-calculus as motivation. Our aim is to give this a precise categorical semantics, to see how far the linear λ-calculus reading can be pushed, to test the resulting predictions empirically, and to be honest about where the account breaks down.

**Contributions.**

1. We decompose the informal claim "*ICL represents higher-order functions; apply/eval are realized in the Transformer; specific heads carry FV-like functionality*" into four independent propositions and identify which the FV literature establishes and which require categorical work (§2).
2. We describe the ambient category of a Transformer layer as a composite of three structures with distinct copy behaviour (§3).
3. Under a linear approximation, we type the FV as a point of a parameter object, provide the realization map bridging the $\mathbb{R}^d$ / $\mathbb{R}^{d\times d}$ dimension gap, and exhibit ICL as a single linear λ-term (§4).
4. We separate three notions of copy and argue that the exponential $!$ is absent: the model has additive and Markov copy but no unbounded multiplicative reuse (§5).
5. We report controlled experiments locating the copy structure in softmax and the applied value in the final MLP (§6).
6. We show that learning is not a natural transformation but a $\mathrm{Para}$ 2-cell, requiring CRDC to give it differential meaning, and separate the parameter and state Jacobians (§7).
7. We position dynamical bifurcation in CRDC and analyze the reducibility of the required extra structure to basic categorical vocabulary (§8), and survey implementations (§9).

Throughout, the epistemic status of claims is marked: some are theorems of the cited literature, some are our constructions under an explicit linear approximation, some are conjectures, and some are empirical findings on a deliberately small model.

## 2. Prior categorical frameworks and the claim, decomposed

Four strands of prior work are relevant, of differing maturity.

- **Categorical Deep Learning** [Gavranović et al. 2024] treats architectures as (co)algebras over (co)monads via polynomial functors — a *generative* foundation rather than a behavioural analysis.
- **Self-Attention as a Parametric Endofunctor** [O'Neill et al. 2025] formalizes self-attention as a parametric 1-morphism in $\mathrm{Para}(\mathbf{Vect})$ whose iterated composition yields a free monad on the layer endofunctor, restricted to the *linear* components (softmax and layer normalization are deferred).
- **The Cognitive Categorical Transformer** uses sheaves, coalgebras, the Yoneda lemma and simplicial complexes as inductive biases — a design proposal rather than an analysis of trained models.
- **Function Vectors** [Todd et al. 2024] is empirical/mechanistic and supplies the λ-calculus vocabulary but constructs no SMCC, internal hom, or evaluation morphism.

The informal eval–apply claim is really four independent propositions:

- **(A, empirical)** Transformers exhibit function-mapping and vector-arithmetic behaviour.
- **(B, modeling)** This can be written in a SMCC such as $\mathbf{Vect}$.
- **(C, representability)** Because SMCC corresponds to (linear) λ-calculus, apply/eval/higher-order structure is representable.
- **(D, localization)** Specific heads carry FV-like functionality.

The FV literature establishes (A) and (D) rigorously; its λ-calculus references are motivational and do not establish (B) or (C). The endofunctor paper reinforces the *form* needed for (B–C) — a parametric morphism with a parameter object — but crucially via the **parameter object of $\mathrm{Para}(\mathbf{Vect})$, not the free monad**, since the free monad models iterated composition of a *fixed* map whereas eval applies a *variable* function. The load-bearing gap is that $\mathbf{Vect}$ is a SMCC but **not** a cartesian closed category (CCC): it has evaluation for linear maps but no $\otimes$-diagonal, so the correspondence is with *linear* λ-calculus, not full simply-typed λ-calculus.

## 3. The ambient category of a Transformer layer

A single layer, taken seriously with softmax and MLP, does not live in one SMCC. It factors through three structures:

$$
\underbrace{\mathrm{Kl}(D)}_{\text{softmax: Markov category}}
\;\xrightarrow{\;D\text{-algebra (expectation)}\;}\;
\underbrace{(\mathbf{Vect},\otimes,\multimap)}_{\text{value mixing / eval}}
\;\xrightarrow{\;\Phi=\text{MLP}\;}\;
\underbrace{(X\multimap Y)}_{\text{apply / realization}}
\;+\;
\underbrace{(\mathbf{Vect},\oplus)}_{\text{residual: additive}}
$$

**Softmax as a Markov kernel.** The attention matrix $A=\mathrm{softmax}(QK^\top)$ has rows that are probability distributions over key positions, i.e. $A$ is a morphism in the Kleisli category of the (finitary) distribution monad $D$, equivalently a morphism in the Markov category $\mathrm{FinStoch}$. The monad $D$ is the *generator* of this Markov category: $\mathrm{Kl}(D)$ is a Markov category precisely because $D$ is a commutative affine monad ($D(1)\cong 1$ gives the unique discarding map; commutativity gives the symmetric monoidal structure). Value mixing $A\,V$ is the action of a $D$-algebra $\mathbb{E}:D(V)\to V$ (convex combination), so a head is a Markov kernel followed by an expectation — an evaluation "into a distribution, then take its mean". This aligns with the Kolmogorov picture in which random variables are functions on a sample space and expectation is a linear functional.

**Value/eval in the SMCC.** In $(\mathbf{Vect},\otimes)$ the internal hom is $A\multimap B=\mathrm{Hom}_k(A,B)$ with counit $\mathrm{ev}:(A\multimap B)\otimes A\to B$, a genuine linear (bilinear-from-the-tensor) map. Evaluation is thus available for linear maps, but there is no natural diagonal $\Delta:A\to A\otimes A$, which forces linearity.

**MLP and residual.** The MLP is best read not as evaluation but as the (non-linear) realization map $\Phi$ that turns an FV point into a function; empirically it behaves as a key–value memory that writes into the residual stream [Geva et al. 2021, 2022]. The residual connection is an additive operation $x\mapsto x+f(x)=(\mathrm{id}+f)(x)$ in the biproduct structure $(\mathbf{Vect},\oplus)$.

## 4. Function Vectors as points, and ICL as a linear λ-term

### 4.1 Typing the FV

In the reference implementation, an FV is computed as a sum over top heads of the out-projection of head activations at the last token, yielding an element of $\mathbb{R}^{d}=\mathbb{R}^{\mathrm{resid\_dim}}$. It is therefore literally a **point** of a parameter object $P_{FV}\subset\mathbb{R}^d$, *not* an element of the internal hom $X\multimap Y\cong\mathbb{R}^{d\times d}$. This dimension gap must be bridged by an explicit **realization map** $\Phi:P_{FV}\to(X\multimap Y)$.

The task index $t$ is **not** a parameter but *data*: it is inferred from the context, not a free weight. The weights $W_{q,k,v,O}$ are shared across all tasks and therefore cannot by themselves select $t$; only the context can. We thus separate two morphisms:

- **Extraction** $E:\mathrm{Ctx}\to P_{FV}$, reading the in-context examples once and producing the point $v_t=E(\mathrm{ctx}_t)$; the weights are the *static* parameters of $E$.
- **Application** $A:P_{FV}\otimes X\to Y$, defined via $\Phi$ and $\mathrm{ev}$.

Robustness of the FV to the choice of insertion site and head set is evidence that the conserved quantity is the point $v_t$ (a site-independent datum), supporting the *point* typing over a position-dependent section.

### 4.2 The linear approximation and MILL

To reach linear λ-calculus we must keep the *bilinear* value path while treating the softmax pattern as frozen (a query-independent constant), because a fully linearized forward pass collapses the FV intervention to an affine translation and destroys $\multimap$. Under this approximation the ambient SMCC is $(\mathbf{Vect}_k,\otimes,I)$ with internal language the multiplicative intuitionistic fragment MILL $(\otimes,\multimap,I)$. The absence of a $\otimes$-diagonal is exactly the linearity discipline "each resource used once".

### 4.3 ICL as a single linear λ-term

With $E$ and $\Phi$ as constants, and using the currying isomorphism $\mathrm{Hom}(C\otimes A,B)\cong\mathrm{Hom}(C,A\multimap B)$:

$$
\dfrac{c:\mathrm{Ctx}\vdash \Phi(E\,c):X\multimap Y \qquad x:X\vdash x:X}
{c:\mathrm{Ctx},\,x:X\vdash (\Phi(E\,c))\,x:Y}\;(\multimap\text{-elim}=\mathrm{ev})
$$

$$
\boxed{\;\lambda c.\,\lambda x.\,(\Phi(E\,c))\,x\;:\;\mathrm{Ctx}\multimap(X\multimap Y)\;}
$$

Because $\multimap$-elimination merges disjoint contexts, both $c$ and $x$ are used exactly once, so the term is well-typed in linear λ-calculus. The reading is: the context is compiled to a *function-typed value* (the type $\mathrm{Ctx}\multimap(X\multimap Y)$), the FV $E(c)$ is a reified **function reference** (a first-class datum), $\Phi$ resolves the reference into a procedure, and $\mathrm{ev}$ applies it. This makes precise the FV literature's statement that the vector *triggers* rather than *performs* the task: $\text{FV}=E(c)$ is the point; $\text{procedure}=\Phi(E(c))\in X\multimap Y$.

An honest caveat: $\Phi$ is an additional, non-automatic structure; whether the downstream effect of an FV is genuinely a multiplicative modulation (eval) rather than a constant additive bias is an empirical question, addressed in §6.

## 5. Copy, revisited: three notions and the absence of $!$

Confusion about "copy" dissolves once three notions are separated.

1. **Multiplicative diagonal** $\Delta:A\to A\otimes A$ (forbidden in $\mathbf{Vect}$; linearity, "use once"). Present *nowhere* in the linear value path.
2. **Additive/biproduct diagonal** $\Delta^{\oplus}:A\to A\oplus A$, $a\mapsto(a,a)$ (always available, linear). This is what the **residual stream** uses: fan-out to sublayers and write-back are $\Delta^{\oplus}$ and $\nabla^{\oplus}$. Residuals recover an *additive* copy along the depth axis; they do **not** relax the once-only discipline of $\mathrm{ev}$, because the two branches are summed back into a single $\otimes$-resource rather than being fed independently into evaluation.
3. **Markov (non-natural) copy** in $\mathrm{Kl}(D)$: every object has a comonoid copy, but it is **not natural**. Non-naturality is not weakness; it is the positive fact that *copy-then-randomize* differs from *randomize-then-copy* — copy generates correlation. This is where **softmax** sits.

None of these is the **exponential modality** $!$ (unbounded, independent reuse). The residual sums its branches (no independent consumption); the Markov copy lacks the naturality that $!$ requires. Consequently the model has **no capacity for unbounded copy**: it is essentially resource-linear at the evaluation site. A "powered/closed Markov category" is the wrong single home, because the internal hom (needed for eval) comes from the $\mathbf{Vect}$ side, not from the Markov side, which is generally not closed. The eval–apply loop must therefore be read as a *composite* of two categories, not one closed Markov category.

This scoping connects to the polynomial-extension view $\mathcal{K}[x]$: adjoining an indeterminate along the residual stream stays at the level of $!$-free *linear* polynomials — the residual carries $x$ across depth but does not multiply it in a $\otimes$-context.

## 6. Empirical probes

We test the structural claims on a small decoder Transformer trained on an in-context permutation-application task (infer a random bijection $\pi$ from $k$ examples, apply it to a query). This is a methodology demonstrator, not a large LM; the same probes attach to real models via forward hooks. Three probes were implemented and run.

**Probe A — copy localization (cross-position mixed second difference).** Perturbing the sublayer-local input at two distinct source positions $a\neq b$ and reading the mixed term $f(+a,+b)-f(+a)-f(+b)+f()$ at the query position isolates each sublayer's own cross-position (i.e. $\otimes$/copy) interaction. A position-wise MLP generates none of its own. Result: the MLP cross-position term is **machine zero** at every layer, while attention is nonzero (ratio $\sim 10^5$–$10^6$). This locates the copy/diagonal structure in softmax attention, not in the pointwise MLP nonlinearity, confirming the structural prediction of §5.

**Probe B — apply localization (activation patching).** Patching the clean run's activation at the query position into a corrupted run (different bijection, same query) shows which component transfers the *applied answer*. Result: mid-layer **attention-out** patching flips the answer (100%) while final-layer **MLP-out** patching flips it (100%); last-layer attention does not. This supports a division of labour: evaluation/routing is carried by attention, and the applied value is written by the final MLP (realization $\Phi$).

**Probe C — reuse ablation (copy = one function, many arguments).** For multi-query prompts, freezing attention to a uniform pattern collapses per-query accuracy to chance, whereas linearizing the MLP (removing the nonlinearity) also degrades accuracy but via the value computation. Result: reuse of one inferred function across queries depends critically on attention routing (the Markov copy of §5), while the MLP nonlinearity is required for the applied value.

Taken together, the probes support **eval = attention (with the copy), apply/realization = MLP**, and reject either monolithic claim ("apply is entirely MLP" or "everything is attention"). We additionally provide an FV-native test suite built on the Function Vectors codebase: an edit-layer sweep testing reuse+realization against a bias-only null, a downstream attention-vs-MLP ablation localizing $\Phi$, and a cross-position interaction probe, to be run on real LMs.

## 7. Learning is not a natural transformation

### 7.1 The negative result

Modelling parameter settings as functors $T_\theta,T_{\theta'}:\mathcal{C}\to\mathcal{D}$, learning $\theta\to\theta'$ is **not** a natural transformation $T_\theta\Rightarrow T_{\theta'}$: there is no family of components $\eta_A:T_\theta A\to T_{\theta'} A$ living in the target category and satisfying naturality; $T_\theta A$ and $T_{\theta'}A$ are different points of $\mathcal{D}$ with no canonical connecting morphism, and the naturality square fails because $T_\theta$ is highly non-linear in $\theta$. This holds for the *simplest linear model* $T_\theta(x)=Wx$, so it is **architecture-independent**: attention does not "break" naturality; it never held. CNN-only, MLP-only, RNN — all the same.

### 7.2 The correct type

Learning is a **2-cell (reparametrisation)** in the 2-category $\mathrm{Para}(\mathbf{Vect})$: a morphism of parameter objects preserving the forward map. A single gradient step is an endo-2-cell $\alpha_\theta:P\to P$; the continuous-time gradient flow is the integral curve of a vector field on $P$. FV extraction internalizes this as a *data-dependent* reparametrisation. Natural transformations *do* appear elsewhere — as the unit/counit of the $D$-free/forgetful adjunction (the evaluation counit), and as symmetries (FFN permutation as a natural isomorphism) — but not as learning.

### 7.3 Two Jacobians

Giving "integral curve" and "Jacobian" meaning requires adding differential structure to $\mathrm{Para}$. In a CRDC, the differential combinator has two restrictions:

- **Parameter Jacobian** $R_P[f]$ (reverse derivative in the parameter argument): drives learning; the loss-Hessian structure whose degeneracy marks saddle-to-saddle transitions (the eNTK view).
- **State Jacobian** $D_A[f]$ (forward derivative in the state argument): its spectrum along trajectories is what Lyapunov analysis measures.

These are projections of one combinator to different arguments — *related* (same $D[f]$) but *distinct* (different arguments, generally independent). This gives a type-level basis for the empirical observation that eNTK cliffs (parameter/data space) and Lyapunov staircases (state space) are related but distinct, and it explains why grokking appears as a saddle-to-saddle phenomenon in the state Jacobian rather than as a natural-transformation-level symmetry.

## 8. Bifurcation in CRDC, and the reducibility of added structure

### 8.1 CRDC

A **Cartesian differential category** (CDC) equips a Cartesian left-additive category with a differential combinator $D[f]:A\times A\to B$ (basepoint, direction) satisfying chain rule, linearity in the direction, and symmetry of second derivatives; in $\mathbf{SMOOTH}$, $D[F](x,v)=J_F(x)\,v$. A **reverse derivative category** (RDC) [Cockett et al. 2020] adds a reverse combinator $R[f]:A\times B\to A$; in $\mathbf{SMOOTH}$, $R[F](x,y)=J_F(x)^{\top}y$ — backpropagation. The central result: a reverse derivative is equivalent to a forward derivative *plus a dagger structure on the subcategory of linear maps*, and those linear maps form an additively enriched category with dagger biproducts. Gradient-based learning is then $\mathrm{Para}+\mathrm{Lens}+\mathrm{RDC}$ [Cruttwell et al. 2022].

### 8.2 Positioning bifurcation

Dynamical systems live naturally in **tangent categories**: a functor $T$ with natural transformations (projection, zero, addition, symmetry, vertical lift) and a universality condition; vector fields are sections of $T$, integral curves solve $\dot\gamma = v\circ\gamma$. A **bifurcation** is a qualitative change in the phase portrait of a parametrized vector field as a parameter crosses a critical value, and it is *located* by the degeneracy of the linearization $D[f]$ — loss of hyperbolicity, an eigenvalue reaching the imaginary axis, i.e. $J$ ceasing to be invertible in the linear (dagger-biproduct) subcategory. Thus CRDC gives a language to *locate* bifurcations. Their *classification* (saddle-node, Hopf, normal forms, topological conjugacy), however, depends on spectral and topological facts that the axioms do not internalize. Honestly: there is at present essentially no categorical bifurcation theory; the differential-category and categorical-dynamical-systems literatures supply the two halves that have not been joined.

### 8.3 Reducibility to basic categorical vocabulary

Whether the extra structure reduces to categories/functors/natural transformations, in the same sense that 2-categories are $\mathbf{Cat}$-enriched and $\infty$-categories are $\mathbf{sSet}$-enriched:

| Added structure | Basic-vocabulary formulation | Ground required externally | Reducibility |
|---|---|---|---|
| Differential | Tangent category: functor $T$ + natural transformations $(p,0,+,c,l)$ + limit preservation | (none essential) | Fully internal |
| Spectral | Dagger category + biproducts (contravariant $\dagger$, natural isos) | Scalar object $\Bbbk=\mathbb{C}$, algebraically closed and complete | Structure reducible; eigenvalue *content* is ground |
| Topological equivalence | $\mathbf{Top}$-enrichment + preservation of the $\mathbb{R}$-action (flow) | Base $\mathbf{Top}$ (or condensed) and a time object | Only via enrichment; not internalized |

So all three are expressible with categories, functors and natural transformations — but, exactly as for 2- and $\infty$-categories, spectral theory and topological equivalence require *choosing an enrichment base* ($\mathbb{C}$-linear complete dagger; $\mathbf{Top}$). Only the differential layer is purely internal (tangent categories). The minimal categorical setting for the bifurcation program is therefore: tangent category (differential, internal) + $\mathbb{C}$-linear complete dagger-biproduct enrichment (spectral, chosen ground) + $\mathbf{Top}$-enrichment or flow $\mathbb{R}$-action (topological, external). The most tractable route is to phrase grokking's saddle-to-saddle as a loss of invertibility / imaginary-axis crossing of the state Jacobian $D_A[f]$ in the dagger subcategory, which uses only the first two layers; topological conjugacy failure then follows via Hartman–Grobman.

## 9. Implementations

Three ecosystems instantiate different parts of this program.

- **catgrad** (Wilson et al.) is the direct CRDC framework: models are open hypergraphs (morphisms of a free symmetric monoidal category), autodiff is a source-to-source transformation via reverse derivatives, and both forward and backward passes compile to static code with no runtime autograd. Predecessors include *Reverse Derivative Ascent* (learning boolean circuits at the level of reverse derivative categories) and lens-based libraries. This is the natural substrate for extracting a *symbolic* Jacobian to feed a spectral/bifurcation analysis.
- **hasktorch** provides Haskell bindings to libtorch with typed and untyped tensors and type-level shape checking. Its differentiation is libtorch's autograd, so it is **not** a CRDC framework, but it is a congenial substrate on which categorical AD abstractions (à la "backprop as functor" / "the simple essence of automatic differentiation") could be built.
- **TorchLean** (arXiv:2602.22631) is a Lean 4 framework for neural-network specification, execution and **verification** (IBP, CROWN, IEEE-754 semantics, autograd-correctness proofs). It is not CRDC-based, but it includes dynamical-system specifications and Lyapunov-style controller verification, making it the relevant tool if one wants to *prove* spectral/stability properties rather than compute them categorically.

No existing tool joins CRDC with dynamical bifurcation; the program of §8 sits precisely at that junction, most plausibly realized by pairing catgrad's reverse-derivative core (symbolic Jacobians) with a spectral/Lyapunov verification layer.

## 10. Conclusion and open problems

We have argued that the eval–apply reading of Transformers is best formalized not in a single SMCC but as a composite of a Markov category (softmax), a symmetric monoidal *closed* category (value/eval), and an additive/non-linear layer (residual/MLP realization). Under a linear approximation the Function Vector is a point of a parameter object, realized by a map $\Phi$ into the internal hom and applied by the evaluation counit, so that one forward pass of ICL is a single linear λ-term. Copy exists in two legitimate but distinct forms — additive (residual) and Markov (softmax) — neither of which supplies the exponential $!$; the model is resource-linear at the evaluation site. Empirically, the copy structure is localized in softmax and the applied value in the final MLP. Learning is not a natural transformation but a $\mathrm{Para}$ 2-cell whose differential meaning requires a CRDC, within which the parameter and state Jacobians separate the learning dynamics from the state dynamics. Bifurcation is *locatable* in CRDC via degeneracy of the state Jacobian in the linear dagger subcategory, but its classification requires an enrichment base for spectra and topology.

**Open problems.** (i) Replace the frozen-softmax linear approximation with a treatment that keeps the Markov structure, and determine whether the non-natural Markov copy suffices for any genuine multi-query reuse. (ii) Determine which component actually realizes $\Phi$ on real LMs (MLP vs attention $QK^\top$ currying), using the FV-native tests. (iii) Formalize grokking's saddle-to-saddle transition as an imaginary-axis crossing of the state Jacobian in a $\mathbb{C}$-linear dagger subcategory, and connect it, via the shared differential combinator, to the parameter-Jacobian (eNTK) transitions. (iv) Build the missing bridge between CRDC and categorical dynamical systems by pairing catgrad's symbolic reverse derivatives with a spectral/Lyapunov layer.

---

## References

Blute, R., Cockett, J.R.B., Seely, R.A.G. (2006). *Differential categories.* Math. Struct. Comput. Sci. 16(6), 1049–1083.

Blute, R., Cockett, J.R.B., Seely, R.A.G. (2009). *Cartesian differential categories.* Theory Appl. Categ. 22(23), 622–672.

Blute, R., Cockett, J.R.B., Lemay, J.-S.P., Seely, R.A.G. (2020). *Differential categories revisited.* Appl. Categ. Struct. 28, 171–235. arXiv:1806.04804.

Cockett, J.R.B., Cruttwell, G.S.H. (2014). *Differential structure, tangent structure, and SDG.* Appl. Categ. Struct. 22(2), 331–417.

Cockett, R., Cruttwell, G., Gallagher, J., Lemay, J.-S.P., MacAdam, B., Plotkin, G., Pronk, D. (2020). *Reverse derivative categories.* CSL 2020, LIPIcs. arXiv:1910.07065.

Cruttwell, G.S.H., Gavranović, B., Ghani, N., Wilson, P., Zanasi, F. (2022). *Categorical foundations of gradient-based learning.* ESOP 2022, LNCS 13240, 1–28. arXiv:2103.01931.

Cruttwell, G., Gallagher, J., Lemay, J.-S.P., Pronk, D. (2022). *Monoidal reverse differential categories.* Math. Struct. Comput. Sci. 32. arXiv:2203.12478.

Cruttwell, G.S.H., Lemay, J.-S.P. (2024). *Reverse tangent categories.* CSL 2024, LIPIcs. arXiv:2308.01131.

Ehrhard, T., Regnier, L. (2003). *The differential λ-calculus.* Theor. Comput. Sci. 309(1), 1–41.

Elliott, C. (2018). *The simple essence of automatic differentiation.* ICFP 2018. arXiv:1804.00746.

Fong, B., Spivak, D.I., Tuyéras, R. (2019). *Backprop as functor: a compositional perspective on supervised learning.* LICS 2019.

Fritz, T. (2020). *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics.* Adv. Math. 370. arXiv:1908.07021.

Gavranović, B. (2024). *Fundamental components of deep learning: a category-theoretic approach.* PhD thesis. arXiv:2403.13001.

Gavranović, B., Lessard, P., Dudzik, A., von Glehn, T., Araújo, J.G.M., Veličković, P. (2024). *Categorical deep learning: an algebraic theory of all architectures.* ICML 2024. arXiv:2402.15332.

Geva, M., Schuster, R., Berant, J., Levy, O. (2021). *Transformer feed-forward layers are key-value memories.* EMNLP 2021, 5484–5495. arXiv:2012.14913.

Geva, M., Caciularu, A., Wang, K., Goldberg, Y. (2022). *Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space.* EMNLP 2022, 30–45.

Jacobs, B. (2017). *Introduction to Coalgebra: Towards Mathematics of States and Observation.* Cambridge University Press.

Jia, Y., Peng, G., Yang, Z., Chen, T. (2025). *Category-theoretical and topos-theoretical frameworks in machine learning: a survey.* Axioms 14(3). arXiv:2408.14014.

Myers, D.J. (2021). *Double categories of open dynamical systems.* EPTCS 333, 154–167. arXiv:2005.05956. (See also *Categorical Systems Theory*, book project.)

O'Neill, C., et al. (2025). *Self-attention as a parametric endofunctor: a categorical framework for Transformer architectures.* arXiv:2501.02931.

Schultz, P., Spivak, D.I., Vasilakopoulou, C. (2020). *Dynamical systems and sheaves.* Appl. Categ. Struct. 28, 1–57. arXiv:1609.08086.

Shiebler, D., Gavranović, B., Wilson, P. (2021). *Category theory in machine learning.* arXiv:2106.07032.

Spivak, D.I. (2020). *Poly: an abundant categorical setting for mode-dependent dynamics.* arXiv:2005.01894.

Todd, E., Li, M.L., Sharma, A.S., Mueller, A., Wallace, B.C., Bau, D. (2024). *Function vectors in large language models.* ICLR 2024. arXiv:2310.15213.

Wilson, P., Zanasi, F. (2021). *Reverse derivative ascent: a categorical approach to learning boolean circuits.* EPTCS 333, 247–260. arXiv:2101.10488.

*Software:* catgrad (hellas-ai/catgrad); hasktorch (hasktorch/hasktorch); TorchLean (lean-dojo/TorchLean, arXiv:2602.22631); Function Vectors (ericwtodd/function_vectors).

---

*Note on epistemic status.* Sections 3–5 are constructions under an explicit linear approximation; §6 reports experiments on a small synthetic model intended as a methodology demonstrator, not as evidence at LM scale; §7's negative result is a theorem-level observation; §8's bifurcation positioning identifies an open gap rather than a completed theory. Claims marked as conjectures should be read as such.
