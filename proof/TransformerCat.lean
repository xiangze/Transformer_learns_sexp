/-
  TransformerCat.lean
  ===================================================================
  A Lean 4 / Mathlib *formalization blueprint* for three categorical claims
  about a single Transformer layer:

    (1) the softmax/Markov routing is a Kleisli morphism of a monad, and the
        whole layer is a Kleisli morphism of a COMPOSITE monad M = T ∘ D
        (obtained from a distributive law of D over T);
    (2) eval-apply is the unit/counit of the tensor–hom adjunction, and the
        in-context term  λc. λx. (Φ (E c)) x  denotes the morphism
        `uncurry (E ≫ Φ)`, i.e. "build the function, then evaluate";
    (3) the residual connection x ↦ x + f(x) is the biproduct diagonal/
        codiagonal Δ_⊕ ≫ (id ⊞ f) ≫ ∇_⊕, an ADDITIVE copy that does NOT
        provide the exponential modality ! .

  HONESTY / STATUS
  ----------------
  * This file is machine-checked against the configured Mathlib version.
  * The SMCC denotation and additive-copy identities are proved.
  * Because the original distributive-law structure omitted Beck's coherence
    equations, it now honestly carries the resulting composite monad as data.
  * The original tensor-diagonal non-existence claim was false (zero maps give
    a counterexample); it is replaced by the valid normalized-discarding
    obstruction below.
-/

import Mathlib.CategoryTheory.Monoidal.Closed.FunctorCategory.Basic
import Mathlib.CategoryTheory.Monoidal.Closed.Basic
import Mathlib.CategoryTheory.Monoidal.Braided.Basic
import Mathlib.CategoryTheory.Limits.Shapes.BinaryBiproducts
import Mathlib.CategoryTheory.Preadditive.Biproducts
import Mathlib.CategoryTheory.Limits.Shapes.IsTerminal
import Mathlib.CategoryTheory.Monad.Kleisli
import Mathlib.CategoryTheory.Monad.Basic
import Mathlib.CategoryTheory.Adjunction.Basic

open CategoryTheory CategoryTheory.MonoidalCategory CategoryTheory.Limits

noncomputable section
namespace TransformerCat

universe v u
variable {C : Type u} [Category.{v} C]

/-! ===================================================================
    CLAIM 3.  Residual = additive (biproduct) copy; not `!`.
    This is the most concretely formalizable claim, so we do it first.
    =================================================================== -/
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


/-! ===================================================================
    CLAIM 2.  eval-apply = counit of the tensor–hom adjunction;
    the ICL term denotes `uncurry (E ≫ Φ)`.
    =================================================================== -/
section EvalApply
variable [MonoidalCategory C] [MonoidalClosed C]
variable {Ctx P X Y : C}

/- Extraction E : Ctx ⟶ P_FV and realization Φ : P_FV ⟶ (X ⊸ Y).
   Here `(ihom X).obj Y` is the internal hom X ⊸ Y. -/
variable (E : Ctx ⟶ P) (Φ : P ⟶ (ihom X).obj Y)

/-- The reified, realized function  f_t = Φ ∘ E : Ctx ⟶ (X ⊸ Y).
    This is the (linear) λ-abstraction / "choose" morphism. -/
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
  The term  λc. λx. (Φ (E c)) x  of linear λ-calculus, of type
  Ctx ⊸ (X ⊸ Y), DENOTES `curriedFn E Φ : Ctx ⟶ (ihom X).obj Y`, and its
  applied form denotes `applyMor E Φ`. Theorems `apply_eq_build_then_ev`, `beta`,
  `eta` are the semantic (SMCC) counterparts of the term's typing plus β/η.

  A *syntactic* soundness theorem — "this linear-λ term is well-typed with each
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


/-! ===================================================================
    CLAIM 1.  The softmax routing is a Kleisli morphism of a monad D,
    and the whole layer is a Kleisli morphism of a composite monad M = T ∘ D.
    =================================================================== -/
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


/-! ===================================================================
    Instance remark.
    The abstract setting is realized by `C = ModuleCat k`:
      * `MonoidalCategory (ModuleCat k)`, `SymmetricCategory`, `MonoidalClosed`
        (tensor ⊗_k, internal hom = k-linear maps) — Claim 2;
      * `Preadditive (ModuleCat k)`, `HasBinaryBiproducts` (⊕ = direct sum)
        — Claim 3;
      * the distribution monad lifts to a monad D on a suitable category for
        Claim 1 (the distributive law with the linear part is the open piece).
    Under this instance `no_natural_discard_of_nonterminal_unit` applies once
    one supplies that the tensor unit `k` is not terminal in `ModuleCat k`
    (for a nontrivial ring).
    =================================================================== -/

end TransformerCat