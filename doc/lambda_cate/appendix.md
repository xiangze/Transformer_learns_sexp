# Appendix
## Appendix A formal proofs of theorems in lean
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
## Appendix B: The relation between distribution moand D, Kleisli categgory $\mathrm{Kl}(D)$ and D-algebra category $\mathrm{EM}(D)$
### The Distribution Monad $D$

Let the base category be $\mathbf{Set}$, and define the finite-support version (the continuous version is discussed later):
$$D(X)=\Big\{\,p:X\to[0,1]\ \Big|\ \mathrm{supp}(p)\text{ finite},\ \textstyle\sum_{x}p(x)=1\,\Big\}$$

This is the set of finitely-supported probability distributions on $X$. The three-part package that makes it a monad:

- **Unit** $\eta_X:X\to D(X)$, $x\mapsto \delta_x$ (the Dirac measure, point mass). "Regard a definite value as a distribution."
- **Multiplication** $\mu_X:D(D(X))\to D(X)$, collapsing a distribution of distributions by averaging: $\mu(P)(x)=\sum_{q}P(q)\,q(x)$. This is exactly the **law of total probability**.
- **Functorial action** $D(f):D(X)\to D(Y)$ (for $f:X\to Y$) is the **pushforward**: $D(f)(p)(y)=\sum_{x:f(x)=y}p(x)$.

The monad laws (left/right unit laws and associativity) coincide with the **basic identities of probability**: the marginalization of Dirac measures and the associativity of mixing. Here $\eta$ is "making definite" and $\mu$ is "flattening a mixture."

**On the base category**: $D$ is **commutative** (the sampling order of two independent distributions does not change the result = Fubini) and **affine** ($D(1)\cong 1$; there is exactly one distribution on a one-point set). These two properties are what later make $\mathrm{Kl}(D)$ a Markov category.

**Variants**: the **Giry monad** $G(X)=\{$probability measures on $X\}$ over measurable spaces $\mathbf{Meas}$ (the continuous version — this is the right one for the real-valued logits of attention); the subdistribution monad for subprobabilities ($\sum\le 1$); and so on. Since softmax has finite support, $D$ suffices.

### The Kleisli Category $\mathrm{Kl}(D)$ — the Category of Stochastic Kernels

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

### $D$-Algebras (the Eilenberg–Moore Category $\mathrm{EM}(D)$)

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

### The Relationship Between $\mathrm{Kl}(D)$ and $\mathrm{EM}(D)$ — Adjunctions and the Comparison Functor

The two are not unrelated; they are two resolutions of the same monad $D$. Every monad arises from an adjunction $F\dashv U$, and $D$ has two canonical ones.

- **Kleisli resolution**: $F_K:\mathbf{Set}\to\mathrm{Kl}(D)$ (free) $\dashv U_K$. $\mathrm{Kl}(D)$ is the category of **free $D$-algebras** only.
- **Eilenberg–Moore resolution**: $F_{EM}:\mathbf{Set}\to\mathrm{EM}(D)$, $X\mapsto(DX,\mu_X)$ (free algebra) $\dashv U_{EM}$ (forgetful, $(X,\alpha)\mapsto X$). $U_{EM}F_{EM}=D$ recovers the monad.

The **counit of $F_{EM}\dashv U_{EM}$ is exactly the algebra structure map** — $\varepsilon_{(X,\alpha)}:(DX,\mu)\to(X,\alpha)$ is $\alpha$ itself. So calling the expectation $\mathbb E$ "counit-like" in the previous group of turns was accurate: **value mixing = a component of the counit of the EM adjunction**.

And the **comparison functor** $K:\mathrm{Kl}(D)\to\mathrm{EM}(D)$, $X\mapsto(DX,\mu_X)$, is fully faithful, embedding $\mathrm{Kl}(D)$ as the **full subcategory of free algebras**:

$$\mathrm{Kl}(D)\ \hookrightarrow\ \mathrm{EM}(D)\qquad(\text{free algebras}).$$

So the two are the "free end ($\mathrm{Kl}$, kernels)" and the "full-algebra end ($\mathrm{EM}$, convex spaces)": the value space $V$ lives in $\mathrm{EM}(D)$ as a **non-free** $D$-algebra, and the attention computation feeds the kernel $A$ of $\mathrm{Kl}(D)$ into the structure map $\alpha_V$ of $V$, an object of $\mathrm{EM}(D)$ — a move that **straddles the two categories**. This is the $D$-side content of "a Transformer is not a single SMCC but a composite of a Markov category and an SMCC."

### Summary Table, and Implications for Unification

| | $\mathrm{Kl}(D)$ | $\mathrm{EM}(D)=\mathrm{Alg}(D)$ |
|---|---|---|
| protagonist (morphism/object) | stochastic kernel (morphism) $X\to D(Y)$ | $D$-algebra (object) $(X,\alpha)$ |
| concretely | stochastic matrix (FinStoch) | convex space / vector space (with expectation) |
| Transformer | **softmax** = kernel $A$ | **value space** $V$, $\alpha_V=\mathbb E$ = value mixing |
| relation to the monad | subcategory of free algebras | full algebras, $U_{EM}F_{EM}=D$ |
| copy | non-natural (generates correlation) | convex structure preserved by affine maps |

The unified category (composite monad $M=T\circ D$)" also becomes visible here. To bundle the linear part $T$ and the distribution $D$ into a single monad, one needs a distributive law $DT\Rightarrow TD$, and its natural candidate is precisely **"expectation commutes with linear maps"** — the very fact that $\alpha_V$ is affine/linear. That the $D$-algebra structure is compatible with the linear structure of $V$ ($\mathbb E$ is linear) is the seed that generates the distributive law between $D$ and $T$. So pinning down the relationship between $\mathrm{Kl}(D)$ and $\mathrm{EM}(D)$ leads directly into the construction of the unified category (the contents of the previous turn's Lean `composeMonad`).

Writing down the distributive law $DT\Rightarrow TD$ from this linearity of expectation = the $D$-algebra is the next move toward unification and Lean formalization. If you want to proceed by concretely defining the candidate for that natural transformation (corresponding to "pushforward of expectation under a linear map," $D(Tf)\to T(Df)$), you can write out each component of the distributive law by taking $V$ as the free vector space monad $T$ and $D$ as the distribution monad.

