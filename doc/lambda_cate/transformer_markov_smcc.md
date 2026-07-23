# Transformers as a Composite of a Markov Category and a Symmetric Monoidal Closed Category

**Abstract.** We give a compositional categorical account of a single Transformer layer. The value path of self-attention is a morphism of the symmetric monoidal closed category (SMCC) $(\mathbf{Vect},\otimes,\multimap)$, whose internal language is multiplicative intuitionistic linear logic (MILL); under a linear approximation this lets us read in-context computation as a **linear λ-term**. The MLP acts as a non-linear realization map and, empirically, as a key–value memory. The residual connection supplies an **additive** copy (the biproduct diagonal), which is a legitimate but weaker notion of copy than the forbidden multiplicative diagonal and than the exponential modality. The softmax attention matrix is a morphism of a **Markov category** — the Kleisli category of the distribution monad — and value mixing is the action of the associated $D$-algebra (expectation). Consequently a Transformer layer is naturally written as a **composite of a Markov category and an SMCC**, plus an additive/non-linear layer, rather than as any single closed category. We ground each component in the existing categorical foundations of machine learning, embed results from a controlled experiment that localizes the copy structure in softmax and the applied value in the MLP, and leave the Function-Vector experiments as clearly marked placeholders to be filled from a real language model.

---

## 1. Thesis and scope

The claim examined here is narrow and structural: a Transformer layer is not modelled by one symmetric monoidal closed category, but by a **composite**

$$
\underbrace{\mathrm{Kl}(D)}_{\text{softmax: Markov category}}
\;\xrightarrow{\;D\text{-algebra (expectation)}\;}\;
\underbrace{(\mathbf{Vect},\otimes,\multimap)}_{\text{value path: SMCC} \;\leftrightarrow\; \text{linear }\lambda\text{-calculus}}
\;\xrightarrow{\;\Phi=\text{MLP}\;}\;
\underbrace{(X\multimap Y)}_{\text{realization / apply}}
\;+\;
\underbrace{(\mathbf{Vect},\oplus)}_{\text{residual: additive copy}} .
$$

Four sub-claims make this up: (i) the attention value path is an SMCC morphism and therefore admits a linear-λ reading (§3); (ii) the MLP is a realization map / key–value memory (§4); (iii) the residual connection realizes a copy, but only in the additive sense (§5); (iv) softmax is a Markov-category morphism (§6). Section 7 assembles the composite, §8 reports experiments, and §9 positions the account against prior work.

## 2. Existing categorical foundations of machine learning

Our decomposition builds on an established body of categorical machine-learning theory, which we use rather than re-derive.

**Gradient-based learning as parametric lenses.** Fong, Spivak and Tuyéras [2019] framed supervised learning functorially ("backprop as functor"). Cruttwell, Gavranović, Ghani, Wilson and Zanasi [2022] gave a categorical semantics of gradient-based learning in terms of *parametrised maps* ($\mathrm{Para}$), *lenses*, and *reverse derivative categories* (RDC), unifying optimisers (SGD, Adam, Nesterov) and losses (MSE, softmax cross-entropy) as instances of a single parametric-lens structure. Elliott [2018] gave the "simple essence" of automatic differentiation as compilation to categories. These works supply the *learning* side of the picture; they are agnostic to the forward architecture.

**Reverse derivative categories.** Cockett et al. [2020] axiomatize the reverse derivative $R[f]:A\times B\to A$ (backpropagation, $J^\top y$ in $\mathbf{SMOOTH}$) and show it equals a forward derivative plus a dagger structure on the subcategory of linear maps, which form an additively enriched category with dagger biproducts. This is the setting in which our two Jacobians (state vs parameter) and the additive biproduct structure of §5 live.

**Architectures as (co)algebras.** Gavranović et al. [2024] ("Categorical Deep Learning") present architectures as (co)algebras over (co)monads via polynomial functors. O'Neill et al. [2025] specifically formalize self-attention as a parametric endofunctor in $\mathrm{Para}(\mathbf{Vect})$, restricted to the linear components — directly the SMCC/value-path fragment we use in §3.

**Probabilistic structure.** Fritz [2020] axiomatizes Markov categories; the Kleisli category of the distribution monad is the canonical example. Shiebler, Gavranović and Wilson [2021], surveying category theory in machine learning, already treat learners as parametrised stochastic processes and distinguish a *co-Kleisli* composition (in which randomness is shared) from a *$\mathrm{Para}$* composition — a precedent for combining probabilistic (Markov) with parametric structure, which is exactly the join we make explicit for attention in §6–7.

**The linear-logic side.** SMCCs are the standard categorical models of multiplicative intuitionistic linear logic; the SMCC ↔ (linear) λ-calculus correspondence is the linear analogue of the Curry–Howard–Lambek correspondence between CCCs and the simply-typed λ-calculus, with roots in the differential/linear-logic tradition (Ehrhard–Regnier). This is what licenses the linear-λ reading in §3.

**Mechanistic anchors.** Geva et al. [2021, 2022] show feed-forward layers act as key–value memories writing into the residual stream (§4). Todd et al. [2024] introduce Function Vectors, compact task representations that *trigger* rather than perform a task, invoking Church's λ-calculus (§3.3, §8.2).

## 3. Self-attention's value path as an SMCC, and the linear-λ reading

### 3.1 The SMCC and its internal language

Work in $(\mathbf{Vect}_k,\otimes,I=k)$, a symmetric monoidal closed category with internal hom $A\multimap B=\mathrm{Hom}_k(A,B)$, tensor–hom adjunction $\mathrm{Hom}(C\otimes A,B)\cong\mathrm{Hom}(C,A\multimap B)$ (currying), and evaluation counit

$$\mathrm{ev}_{A,B}:(A\multimap B)\otimes A\longrightarrow B,$$

which is bilinear, hence a genuine linear morphism from the tensor. This is the categorical content of "evaluation is available for linear maps". Crucially $\mathbf{Vect}$ has **no** natural diagonal $\Delta:A\to A\otimes A$ (the map $a\mapsto a\otimes a$ is non-linear), so copy/delete are absent from the multiplicative fragment. The internal language of this SMCC is MILL $(\otimes,\multimap,I)$: types are objects, terms are morphisms, $\multimap$-introduction is currying, $\multimap$-elimination is $\mathrm{ev}$, and the linearity discipline ("each variable used exactly once") is the categorical absence of the diagonal.

### 3.2 The linear approximation

To keep evaluation non-trivial we freeze the softmax pattern (treat the attention weights as a query-independent constant) while retaining the *bilinear* value path $V=W_v h$ and the query–value contraction. A fully linearized forward pass would collapse an additive intervention to a translation and trivialize $\multimap$; the frozen-pattern approximation instead preserves the counit $\mathrm{ev}$. All λ-typing below is under this approximation; §6 restores the softmax as a Markov morphism.

### 3.3 Function Vectors as points; realization and application

The reference Function-Vector construction sums, over the top heads, the out-projection of head activations at the last token, yielding a vector in $\mathbb{R}^d=\mathbb{R}^{\mathrm{resid\_dim}}$. An FV is therefore a **point** of a parameter object $P_{FV}\subset\mathbb{R}^d$, not an element of the internal hom $X\multimap Y\cong\mathbb{R}^{d\times d}$. Bridging this dimension gap requires a **realization map** $\Phi:P_{FV}\to(X\multimap Y)$. The task index $t$ is data (inferred from context), not a free parameter; we separate extraction and application:

$$E:\mathrm{Ctx}\multimap P_{FV},\qquad A:=\mathrm{ev}\circ(\Phi\otimes\mathrm{id}_X):P_{FV}\otimes X\to Y,\qquad v_t=E(\mathrm{ctx}_t).$$

### 3.4 In-context computation as a linear λ-term

With $E,\Phi$ as constants and using currying, one forward pass of ICL is well-typed in linear λ-calculus:

$$
\dfrac{c:\mathrm{Ctx}\vdash \Phi(E\,c):X\multimap Y \qquad x:X\vdash x:X}
{c:\mathrm{Ctx},\,x:X\vdash (\Phi(E\,c))\,x:Y}\;(\multimap\text{-elim}=\mathrm{ev})
\qquad\Longrightarrow\qquad
\boxed{\;\lambda c.\,\lambda x.\,(\Phi(E\,c))\,x\;:\;\mathrm{Ctx}\multimap(X\multimap Y)\;}
$$

Both $c$ and $x$ are used exactly once, so the term is linear. The type $\mathrm{Ctx}\multimap(X\multimap Y)$ says the context compiles to a *function-typed value*: the FV $E(c)$ is a reified function reference (a first-class datum), $\Phi$ resolves it to a procedure, and $\mathrm{ev}$ applies it — a precise reading of "the vector triggers, but does not perform, the task" [Todd et al. 2024].

## 4. The MLP as realization and key–value memory

The MLP is best read not as evaluation but as the (non-linear) realization map $\Phi:P_{FV}\to(X\multimap Y)$ that converts an FV point into an applied function. This is consistent with the mechanistic result that feed-forward layers act as **key–value memories**: each key correlates with input patterns and each value induces an output distribution, with the layer output a composition of memories refined through the residual stream [Geva et al. 2021, 2022]. In our composite the MLP writes additively into the residual (the $\oplus$ side of §5) while realizing the currying step of §3. It is therefore the natural carrier of $\Phi$, and we test this localization in §8 (Probe B, Probe C).

## 5. The residual connection as additive copy

A residual connection is $x\mapsto x+f(x)=(\mathrm{id}_A+f)(x)$, a sum of parallel morphisms, legitimate because $\mathbf{Vect}$ is additively enriched with biproducts. It uses the **additive/biproduct diagonal** $\Delta^{\oplus}:A\to A\oplus A$, $a\mapsto(a,a)$ (fan-out to sublayers) and codiagonal $\nabla^{\oplus}:A\oplus A\to A$ (write-back), both linear. Three notions of copy must be separated:

| Notion | Structure | Axis | Carrier | Status |
|---|---|---|---|---|
| Multiplicative diagonal | $\Delta:A\to A\otimes A$ | — | none | Forbidden (linearity, "use once") |
| Additive diagonal | $\Delta^{\oplus}:A\to A\oplus A$ | depth (across layers) | residual stream | Legal, linear |
| Markov copy | non-natural comonoid in $\mathrm{Kl}(D)$ | position (across tokens) | softmax | Legal, generates correlation |

So the residual connection *does* recover a copy — the **additive** one, along the depth axis — which is why "copy exists in some sense" is correct. But it is not the multiplicative diagonal, and it is not the exponential modality $!$: the two branches are summed back by $\nabla^{\oplus}$ into a single $\otimes$-resource rather than being consumed independently, so the once-only discipline of $\mathrm{ev}$ is untouched. The residual therefore adds the *additive* fragment $(\oplus,\&)$ to the multiplicative $(\otimes,\multimap)$ without breaking linearity, and does not supply unbounded reuse.

## 6. Softmax as a Markov-category morphism

Each row of $A=\mathrm{softmax}(QK^\top)$ is a probability distribution over key positions. Hence $A$ is a **Markov kernel**: a morphism of the Kleisli category $\mathrm{Kl}(D)$ of the distribution monad $D$ (equivalently, $\mathrm{FinStoch}$), the canonical Markov category [Fritz 2020]. The monad $D$ is the *generator* of this Markov category — $\mathrm{Kl}(D)$ is a Markov category because $D$ is a commutative affine monad ($D(1)\cong 1$ yields the unique discarding map; commutativity yields the symmetric monoidal structure). Value mixing $A\,V$ is the action of the associated **$D$-algebra** $\mathbb{E}:D(V)\to V$ (convex combination / expectation), which aligns with the Kolmogorov picture in which random variables are functions on a sample space and expectation is a linear functional. A head thus decomposes as

$$\text{head}=\Big[\;\mathrm{Ctx}\xrightarrow{\ \text{softmax}\ } D(X)\;\Big]\ \text{followed by}\ \Big[\;D(X)\xrightarrow{\ \mathbb{E}\ } X\;\Big],$$

a Markov kernel composed with the $D$-algebra structure map — the probabilistic generalization of the SMCC evaluation of §3. The Markov copy is *non-natural*: copy-then-randomize differs from randomize-then-copy, so it generates correlation rather than duplicating a value freely; this is the position-axis copy of §5 and, as we note in §7, it is not the exponential $!$.

## 7. The composite: Markov category ∘ SMCC (+ additive/MLP)

Assembling §§3–6, a Transformer layer is the composite displayed in §1: a Markov morphism (softmax) feeding, through the $D$-algebra expectation, the SMCC value path (where evaluation and the linear-λ reading live), then the non-linear realization $\Phi$ (MLP), with the additive residual $(\mathbf{Vect},\oplus)$ threading depth. Three points make the composite non-collapsible into a single closed category.

1. **The closed structure comes from the SMCC side, not the Markov side.** Markov categories are generally not closed; the internal hom that supports $\mathrm{ev}$/apply is $\mathbf{Vect}$'s $\multimap$. A single "closed Markov category" cannot host both the non-natural copy and the closed structure; the correct object is the *composite* of two categories bridged by the $D$-algebra.
2. **Copy differs on each side.** The value/eval SMCC has *no* copy (once-only); the residual has *additive* copy; softmax has *non-natural Markov* copy. None is the exponential $!$, so the whole layer has no unbounded-reuse capacity.
3. **The learning side is orthogonal and already categorified.** Parameter updates are handled by the $\mathrm{Para}$/lens/RDC machinery of the existing foundations [Cruttwell et al. 2022; Cockett et al. 2020]; our contribution is the *forward-architecture* decomposition, extending the linear-endofunctor picture of O'Neill et al. [2025] with the Markov (softmax) and additive (residual) structure, in the spirit of the Kleisli+$\mathrm{Para}$ combination already anticipated by Shiebler et al. [2021].

## 8. Experiments

### 8.1 Controlled probes on a synthetic in-context model

We train a small decoder Transformer ($V=10$ symbols, $k=5$ in-context examples, $d=64$, $L=3$ layers, $4$ heads) on an in-context permutation-application task (infer a random bijection $\pi$ from examples, apply it to queries). This is a methodology demonstrator, not a language model; the same probes attach to real models via forward hooks. The model reaches **single-query accuracy $1.000$** (chance $0.100$).

**Probe A — copy localization (sublayer-local cross-position mixed second difference).** Perturbing a sublayer's own input at two distinct source positions $a\neq b$ and reading $f(+a,+b)-f(+a)-f(+b)+f()$ at the query position isolates each sublayer's own cross-position ($\otimes$/copy) interaction; a strictly position-wise MLP generates none of its own.

| Layer | attention | MLP | ratio attn/mlp |
|---|---|---|---|
| 0 | $5.47\times10^{-3}$ | $0.0$ | $\sim 5\times10^{6}$ |
| 1 | $4.11\times10^{-4}$ | $0.0$ | $\sim 4\times10^{5}$ |
| 2 | $5.07\times10^{-4}$ | $0.0$ | $\sim 5\times10^{5}$ |

The MLP cross-position term is **machine zero** at every layer; only attention carries cross-position interaction. This localizes the copy/diagonal structure in softmax attention, not in the pointwise MLP nonlinearity — the empirical counterpart of §5–6.

**Probe B — apply localization (activation patching, flip-to-clean rate).** Patching the clean run's activation at the query position into a corrupted run (different bijection, same query) shows which component transfers the applied answer (356 usable trials).

| Component | Layer 0 | Layer 1 | Layer 2 |
|---|---|---|---|
| attention-out | $0.00$ | $\mathbf{1.00}$ | $0.00$ |
| MLP-out | $0.00$ | $0.32$ | $\mathbf{1.00}$ |

Mid-layer attention (L1) fully transfers the answer, and the final-layer MLP (L2) fully writes it; last-layer attention does not. This supports **evaluation/routing in attention** and **realization $\Phi$ in the final MLP**, matching §3–4.

**Probe C — reuse ablation (per-query accuracy on multi-query prompts).**

| Condition | q0 | q1 | q2 |
|---|---|---|---|
| intact | $1.000$ | $1.000$ | $1.000$ |
| freeze attention (uniform) | $0.047$ | $0.027$ | $0.090$ |
| linearize MLP (remove nonlinearity) | $0.148$ | $0.166$ | $0.180$ |

Freezing attention collapses multi-query reuse to chance (the Markov copy of §6 is what routes one function to many arguments), while linearizing the MLP degrades the applied value. Both components are necessary, in the two distinct roles predicted: attention carries routing/copy, MLP carries the applied value.

Together the probes support **eval = attention (with the copy), apply/realization = MLP**, and reject either monolithic reading.

### 8.2 Function-Vector experiments (to be filled)

The following experiments run the same three questions on a real language model using the Function-Vector pipeline (`ericwtodd/function_vectors`): FV extraction (mean head activations → indirect effect → summed top heads), intervention via `add_function_vector`, and `baukit.TraceDict` hooks. Results are **left blank pending execution on a GPU environment** (the reference models are downloaded from a source not reachable from the drafting environment).

**FV-1 — reuse + realization (edit-layer sweep, single fixed FV across many queries).**
Model: `__________`. Task: `__________`. FV top heads: `__________`.

| edit layer | FV-insertion zero-shot accuracy |
|---|---|
| … | `_____` |
| $L^{*}$ (peak) | `_____` |
| … | `_____` |
| bias-only null ($W_U\,\mathrm{ln}_f(\mathrm{FV})$) | `_____` |

*Expected reading:* a mid-layer peak $\gg$ bias-only indicates that one fixed point, reused across all queries, is realized+applied downstream ($\Phi$+eval) rather than acting as a constant logit bias.

**FV-2 — apply localization (ablate attention-out vs MLP-out downstream of $L^{*}$).**
Base FV-insertion accuracy: `_____`.

| layer $l>L^{*}$ | attn-ablate drop | mlp-ablate drop |
|---|---|---|
| … | `_____` | `_____` |

*Expected reading:* larger MLP drops localize realization/apply ($\Phi$) in the MLP; larger attention drops localize evaluation in attention.

**FV-3 — copy localization (cross-position interaction, attention vs MLP per layer).**

| layer | attention | MLP | ratio |
|---|---|---|---|
| … | `_____` | `_____` | `_____` |

*Expected reading:* attention-dominant cross-position term corroborates the softmax (Markov) copy of §6, with the toy model providing the exact position-wise isolation.

## 9. Related work and positioning

The *learning* side of Transformers is already on firm categorical footing: parametric lenses and reverse derivative categories [Fong–Spivak–Tuyéras 2019; Cruttwell et al. 2022; Cockett et al. 2020], surveyed in [Shiebler et al. 2021], and the (co)algebraic architecture theory of [Gavranović et al. 2024]. The *forward* structure of attention was categorified for its linear part by O'Neill et al. [2025]. Our account adds the two structures that the linear picture defers: the **Markov** structure of softmax [Fritz 2020] and the **additive** structure of residuals, and states how they compose with the SMCC value path and the MLP realization. The combination of a probabilistic (Kleisli) and a parametric structure was already anticipated in the survey of Shiebler et al. [2021]; we make it concrete for attention and connect it, via the linear-λ reading, to the Function-Vector phenomenology of Todd et al. [2024] and the key–value-memory role of the MLP [Geva et al. 2021, 2022].

## 10. Conclusion

A Transformer layer is faithfully described not by one symmetric monoidal closed category but by a composite: a Markov category (softmax) bridged by a $D$-algebra expectation to an SMCC (value path, where evaluation and a linear-λ reading live), followed by a non-linear realization map (MLP) and threaded by an additive residual copy. Copy exists in two legitimate but distinct forms — additive (residual, depth axis) and Markov (softmax, position axis) — neither of which is the exponential modality, so the layer is resource-linear at the evaluation site. Controlled experiments localize the copy structure in softmax and the applied value in the MLP; the Function-Vector experiments that would confirm this at language-model scale are specified and left as placeholders.

---

## References

Cockett, R., Cruttwell, G., Gallagher, J., Lemay, J.-S.P., MacAdam, B., Plotkin, G., Pronk, D. (2020). *Reverse derivative categories.* CSL 2020. arXiv:1910.07065.

Cruttwell, G.S.H., Gavranović, B., Ghani, N., Wilson, P., Zanasi, F. (2022). *Categorical foundations of gradient-based learning.* ESOP 2022, LNCS 13240, 1–28. arXiv:2103.01931.

Ehrhard, T., Regnier, L. (2003). *The differential λ-calculus.* Theor. Comput. Sci. 309(1), 1–41.

Elliott, C. (2018). *The simple essence of automatic differentiation.* ICFP 2018. arXiv:1804.00746.

Fong, B., Spivak, D.I., Tuyéras, R. (2019). *Backprop as functor: a compositional perspective on supervised learning.* LICS 2019.

Fritz, T. (2020). *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics.* Adv. Math. 370. arXiv:1908.07021.

Gavranović, B., Lessard, P., Dudzik, A., von Glehn, T., Araújo, J.G.M., Veličković, P. (2024). *Categorical deep learning: an algebraic theory of all architectures.* ICML 2024. arXiv:2402.15332.

Geva, M., Schuster, R., Berant, J., Levy, O. (2021). *Transformer feed-forward layers are key-value memories.* EMNLP 2021, 5484–5495. arXiv:2012.14913.

Geva, M., Caciularu, A., Wang, K., Goldberg, Y. (2022). *Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space.* EMNLP 2022, 30–45.

O'Neill, C., et al. (2025). *Self-attention as a parametric endofunctor: a categorical framework for Transformer architectures.* arXiv:2501.02931.

Shiebler, D., Gavranović, B., Wilson, P. (2021). *Category theory in machine learning.* arXiv:2106.07032.

Todd, E., Li, M.L., Sharma, A.S., Mueller, A., Wallace, B.C., Bau, D. (2024). *Function vectors in large language models.* ICLR 2024. arXiv:2310.15213.

*Software:* `ericwtodd/function_vectors`; the synthetic-probe script `eval_apply_probes.py` and the FV-native script `eval_apply_fv.py` accompanying this note.

---

*Epistemic status.* §§3–7 are constructions under an explicit linear approximation (frozen softmax for the λ-typing; the Markov reading of §6 restores it). §8.1 reports a small synthetic model as a methodology demonstrator, not evidence at language-model scale. §8.2 is a specification with results deliberately left blank.
