/-
Copyright (c) 2026 Xiangze. All rights reserved.
Released under Apache 2.0 license.

# ゲージ完全性と実現写像のファイバー

`ρ : Θ → F` を実現写像（パラメータ ↦ ネットワークが計算する関数）、
`G` を `Θ` に作用するゲージ群とし、`ρ` は `G` 不変とする。

このファイルで形式化するもの：

1. **商写像** `ρ̄ : Θ/G → F` の構成（`descend`）。
   `ρ̄` は *常に* `ρ` の像の上へ全射である（`range_descend`）。
   したがって「`Θ//G → F` が全射」は完全性の条件ではなく、
   本当の条件は **`ρ̄` の単射性** である（`complete_iff_injective`）。
2. **作用亜群** `Θ//G`（Mathlib の `ActionCategory G Θ`）から離散圏 `Discrete F` への
   実現関手 `R̄` を作り、**完全性 ⟺ `R̄` が充満（full）** を証明する（`complete_iff_full`）。
3. 完全なファイバーは等質空間 `G ⧸ Stab(θ)` に同型である（`fiberEquivQuotientStabilizer`）。
   特に同じファイバー上の点の固定化群は互いに共役である
   （`stabilizer_conj_of_completeAt`）。
   対偶として **固定化群の跳び**（ゲージ固定点と非固定点が同じファイバーに乗る）は
   不完全性を強制する（`not_completeAt_of_fixed_of_not_fixed`）。
   滑らかな設定では、これがまさに軌道型の層化が変わる点、すなわち商の特異点である。
4. **最小の玩具模型**：幅 1 の QK 結合（あるいは 2 層線形ネット）
   `ρ(q, k) = q·k`、ゲージ `t • (q, k) = (t q, t⁻¹ k)`（`t ∈ Kˣ`）。
   * `c ≠ 0` のファイバーでは完全（`QK.completeAt_of_ne_zero`）、
   * `c = 0` のファイバーでは不完全（`QK.not_completeAt_zero`）、
   * 不完全なファイバーはちょうど臨界点（線形部分が消える点）の像
     （`QK.not_completeAt_iff_mem_critical_image`）、
   * そのファイバーは 2 本の軸の和（正規交叉）（`QK.fiber_zero`）。

形式化して *いない* もの：滑らかな商・軌道型層化・RLCT の定義、
実際の多層 Transformer のゲージ群に対する完全性（未解決）。
-/
import Mathlib.GroupTheory.GroupAction.Quotient
import Mathlib.CategoryTheory.Action
import Mathlib.CategoryTheory.Discrete.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.LinearCombination

namespace GaugeCompleteness

open MulAction

/-! ## 1. 抽象的設定 -/

section Abstract

variable (G : Type*) {Θ F : Type*} [Group G] [MulAction G Θ]

/-- 実現写像 `ρ` がゲージ不変であること。 -/
def GaugeInvariant (ρ : Θ → F) : Prop :=
  ∀ (g : G) (θ : Θ), ρ (g • θ) = ρ θ

/-- ファイバー `ρ⁻¹(f)` 上で `G` が推移的に作用すること。 -/
def CompleteAt (ρ : Θ → F) (f : F) : Prop :=
  ∀ θ θ' : Θ, ρ θ = f → ρ θ' = f → ∃ g : G, g • θ = θ'

/-- `G` が `ρ` に対して **完全**：すべてのファイバーに推移的に作用する。 -/
def Complete (ρ : Θ → F) : Prop :=
  ∀ f : F, CompleteAt G ρ f

variable {G} {ρ : Θ → F}

/-- 商 `Θ/G` から関数空間への誘導写像 `ρ̄`。 -/
def descend (hρ : GaugeInvariant G ρ) : orbitRel.Quotient G Θ → F :=
  Quotient.lift ρ fun a b h => by
    obtain ⟨g, rfl⟩ := (orbitRel_apply.mp h)
    exact hρ g b

@[simp]
theorem descend_mk (hρ : GaugeInvariant G ρ) (θ : Θ) :
    descend hρ (Quotient.mk (orbitRel G Θ) θ) = ρ θ :=
  rfl

/-- `ρ̄` の像は `ρ` の像と一致する。すなわち `Θ/G → im ρ` は **常に全射**。 -/
theorem range_descend (hρ : GaugeInvariant G ρ) :
    Set.range (descend hρ) = Set.range ρ := by
  ext f
  constructor
  · rintro ⟨x, rfl⟩
    induction x using Quotient.inductionOn with
    | h θ => exact ⟨θ, rfl⟩
  · rintro ⟨θ, rfl⟩
    exact ⟨Quotient.mk _ θ, rfl⟩

/-- **完全性 ⟺ `ρ̄ : Θ/G → F` が単射。** -/
theorem complete_iff_injective (hρ : GaugeInvariant G ρ) :
    Complete G ρ ↔ Function.Injective (descend hρ) := by
  constructor
  · intro hc x y hxy
    induction x using Quotient.inductionOn with
    | h a =>
    induction y using Quotient.inductionOn with
    | h b =>
      obtain ⟨g, hg⟩ := hc (ρ a) a b rfl hxy.symm
      apply Quotient.sound
      exact orbitRel_apply.mpr ⟨g⁻¹, inv_smul_eq_iff.mpr hg.symm⟩
  · intro hinj f θ θ' h₁ h₂
    have h : descend hρ (Quotient.mk _ θ) = descend hρ (Quotient.mk _ θ') := by
      simp [h₁, h₂]
    obtain ⟨g, hg⟩ := orbitRel_apply.mp (Quotient.exact (hinj h))
    exact ⟨g⁻¹, inv_smul_eq_iff.mpr hg.symm⟩

/-- 同値な言い換え：`Θ/G → im ρ` が全単射 ⟺ 完全。全射部分は自明なので、
情報はすべて単射性にある。 -/
theorem complete_iff_bijective_onto_range (hρ : GaugeInvariant G ρ) :
    Complete G ρ ↔ Function.Bijective (Set.rangeFactorization (descend hρ)) := by
  rw [Set.rangeFactorization_bijective, complete_iff_injective hρ]

/-- ゲージ不変性の下では、点 `θ` を含むファイバーでの完全性は
「ファイバー = 軌道」と同値。 -/
theorem completeAt_iff_fiber_eq_orbit (hρ : GaugeInvariant G ρ) (θ : Θ) :
    CompleteAt G ρ (ρ θ) ↔ ρ ⁻¹' {ρ θ} = orbit G θ := by
  constructor
  · intro hc
    ext θ'
    simp only [Set.mem_preimage, Set.mem_singleton_iff]
    constructor
    · intro h
      exact mem_orbit_iff.mpr (hc θ θ' rfl h)
    · rintro ⟨g, rfl⟩
      exact hρ g θ
  · intro heq θ₁ θ₂ h₁ h₂
    have m₁ : θ₁ ∈ orbit G θ := by rw [← heq]; exact h₁
    have m₂ : θ₂ ∈ orbit G θ := by rw [← heq]; exact h₂
    obtain ⟨g₁, rfl⟩ := m₁
    obtain ⟨g₂, rfl⟩ := m₂
    refine ⟨g₂ * g₁⁻¹, ?_⟩
    simp [mul_smul]

/-- 完全なファイバーは等質空間 `G ⧸ Stab(θ)`（軌道・固定化群定理）。 -/
noncomputable def fiberEquivQuotientStabilizer (hρ : GaugeInvariant G ρ) (θ : Θ)
    (hc : CompleteAt G ρ (ρ θ)) :
    (ρ ⁻¹' {ρ θ}) ≃ G ⧸ stabilizer G θ :=
  (Set.equivOfEq ((completeAt_iff_fiber_eq_orbit hρ θ).mp hc)).trans
    (orbitEquivQuotientStabilizer G θ)

/-- 完全なファイバー上の 2 点の固定化群は共役。 -/
theorem stabilizer_conj_of_completeAt {θ θ' : Θ}
    (hc : CompleteAt G ρ (ρ θ)) (h : ρ θ' = ρ θ) :
    ∃ g : G, stabilizer G θ' = (stabilizer G θ).map (MulAut.conj g).toMonoidHom := by
  obtain ⟨g, rfl⟩ := hc θ θ' rfl h
  exact ⟨g, stabilizer_smul_eq_stabilizer_map_conj g θ⟩

/-- **固定化群の跳び ⟹ 不完全。**
ゲージ全体で固定される点 `θ₀`（例：`θ₀ = 0`）と固定されない点 `θ₁` が
同じファイバーに乗れば、そのファイバーで `G` は推移的でない。 -/
theorem not_completeAt_of_fixed_of_not_fixed {θ₀ θ₁ : Θ}
    (hfix : ∀ g : G, g • θ₀ = θ₀) (hmove : ∃ g : G, g • θ₁ ≠ θ₁)
    (hfib : ρ θ₁ = ρ θ₀) : ¬ CompleteAt G ρ (ρ θ₀) := by
  intro hc
  obtain ⟨g, rfl⟩ := hc θ₀ θ₁ rfl hfib
  obtain ⟨h, hh⟩ := hmove
  apply hh
  -- `h • g • θ₀ = g • θ₀`：両辺とも `θ₀` の固定性で `θ₀` になる
  simp only [hfix]

end Abstract

/-! ## 2. 作用亜群と実現関手 -/

section Groupoid

open CategoryTheory

variable {G : Type*} {Θ F : Type*} [Group G] [MulAction G Θ] {ρ : Θ → F}

/-- 実現関手 `R̄ : Θ//G ⥤ Discrete F`。 -/
def realizationFunctor (hρ : GaugeInvariant G ρ) : ActionCategory G Θ ⥤ Discrete F where
  obj x := ⟨ρ x.back⟩
  map {x y} φ := Discrete.eqToHom (by
    show ρ x.back = ρ y.back
    have hv := φ.map_val
    simp only [actionAsFunctor_map] at hv
    change φ.hom • x.back = y.back at hv
    rw [← hv, hρ])
  map_id _ := Subsingleton.elim _ _
  map_comp _ _ := Subsingleton.elim _ _

@[simp]
theorem realizationFunctor_obj (hρ : GaugeInvariant G ρ) (θ : Θ) :
    (realizationFunctor hρ).obj (θ : ActionCategory G Θ) = ⟨ρ θ⟩ :=
  rfl

/-- **完全性 ⟺ 実現関手 `Θ//G ⥤ Discrete F` が充満。**
（対象レベルでは `range_descend` により `im ρ` に対し常に本質的全射。） -/
theorem complete_iff_full (hρ : GaugeInvariant G ρ) :
    Complete G ρ ↔ (realizationFunctor hρ).Full := by
  constructor
  · intro hc
    refine ⟨fun {x y} ψ => ?_⟩
    have hxy : ρ x.back = ρ y.back := Discrete.eq_of_hom ψ
    obtain ⟨g, hg⟩ := hc (ρ x.back) x.back y.back rfl hxy.symm
    exact ⟨Functor.Elements.homMk g (by exact hg), Subsingleton.elim _ _⟩
  · intro hfull f θ θ' h₁ h₂
    have ψ : (realizationFunctor hρ).obj (θ : ActionCategory G Θ) ⟶
        (realizationFunctor hρ).obj (θ' : ActionCategory G Θ) :=
      Discrete.eqToHom (by simp [h₁, h₂])
    obtain ⟨φ, -⟩ := (realizationFunctor hρ).map_surjective ψ
    have hv := φ.map_val
    simp only [actionAsFunctor_map] at hv
    exact ⟨φ.hom, hv⟩

end Groupoid

/-! ## 3. 玩具模型：幅 1 の QK 結合 `ρ(q,k) = q k` -/

namespace QK

variable (K : Type*) [Field K]

/-- パラメータ `(q, k)`：1 次元クエリ・キー重み（または 2 層線形ネットの重み）。 -/
@[ext]
structure Param where
  q : K
  k : K

variable {K}

/-- ゲージ作用 `t • (q, k) = (t q, t⁻¹ k)`。QK ゲージ `W_Q ↦ G W_Q, W_K ↦ G^{-⊤} W_K`
の `d_k = 1` 版。 -/
instance : MulAction Kˣ (Param K) where
  smul t p := ⟨t * p.q, (t⁻¹ : Kˣ) * p.k⟩
  one_smul p := by ext <;> simp [HSMul.hSMul, SMul.smul]
  mul_smul s t p := by
    ext <;> simp [HSMul.hSMul, SMul.smul] <;> ring

@[simp] theorem smul_q (t : Kˣ) (p : Param K) : (t • p).q = t * p.q := rfl
@[simp] theorem smul_k (t : Kˣ) (p : Param K) : (t • p).k = (t⁻¹ : Kˣ) * p.k := rfl

/-- 実現写像：ロジット係数 `q k`。 -/
def ρ (p : Param K) : K := p.q * p.k

theorem gaugeInvariant : GaugeInvariant Kˣ (ρ (K := K)) := by
  intro t p
  simp only [ρ, smul_q, smul_k]
  rw [mul_mul_mul_comm, Units.mul_inv, one_mul]

/-- `c ≠ 0` のファイバーではゲージ群は完全（ファイバーは双曲線 = 1 軌道）。 -/
theorem completeAt_of_ne_zero {c : K} (hc : c ≠ 0) : CompleteAt Kˣ ρ c := by
  intro p p' h h'
  simp only [ρ] at h h'
  have hq : p.q ≠ 0 := by rintro hq; rw [hq, zero_mul] at h; exact hc h.symm
  have hq' : p'.q ≠ 0 := by rintro hq; rw [hq, zero_mul] at h'; exact hc h'.symm
  refine ⟨Units.mk0 (p'.q / p.q) (div_ne_zero hq' hq), ?_⟩
  ext
  · simp only [smul_q, Units.val_mk0]
    field_simp
  · simp only [smul_k, Units.val_inv_eq_inv_val, Units.val_mk0, inv_div]
    field_simp
    linear_combination h - h'

/-- `c = 0` のファイバーでは不完全：`(1,0)` と `(0,1)` は同じファイバーにあるが
ゲージで結べない。 -/
theorem not_completeAt_zero : ¬ CompleteAt Kˣ (ρ (K := K)) 0 := by
  intro hc
  obtain ⟨t, ht⟩ := hc ⟨1, 0⟩ ⟨0, 1⟩ (by simp [ρ]) (by simp [ρ])
  have := congrArg Param.q ht
  simp at this

/-- 抽象定理（固定化群の跳び）による別証明：原点は全ゲージで固定され、
`(1,0)` は `t ≠ 1` で動く。 -/
theorem not_completeAt_zero' (hK : ∃ t : Kˣ, t ≠ 1) : ¬ CompleteAt Kˣ (ρ (K := K)) 0 := by
  have h := not_completeAt_of_fixed_of_not_fixed (G := Kˣ) (ρ := ρ (K := K))
    (θ₀ := ⟨0, 0⟩) (θ₁ := ⟨1, 0⟩)
    (fun t => by ext <;> simp)
    (by
      obtain ⟨t, ht⟩ := hK
      refine ⟨t, fun h => ht ?_⟩
      have := congrArg Param.q h
      simp only [smul_q, mul_one] at this
      exact Units.ext this)
    (by simp [ρ])
  simpa [ρ] using h

/-- 完全性の判定：`CompleteAt c ⟺ c ≠ 0`。 -/
theorem completeAt_iff (c : K) : CompleteAt Kˣ ρ c ↔ c ≠ 0 := by
  constructor
  · rintro hc rfl; exact not_completeAt_zero hc
  · exact completeAt_of_ne_zero

/-- `ρ` の厳密なテイラー展開：
`ρ(p + h) = ρ(p) + dρ_p(h) + h_q h_k`、ここで `dρ_p(h) = k h_q + q h_k`。 -/
def dρ (p h : Param K) : K := p.k * h.q + p.q * h.k

theorem taylor (p h : Param K) :
    ρ ⟨p.q + h.q, p.k + h.k⟩ = ρ p + dρ p h + h.q * h.k := by
  simp only [ρ, dρ]; ring

/-- 臨界点：線形部分 `dρ_p` が恒等的に 0。 -/
def IsCritical (p : Param K) : Prop := ∀ h, dρ p h = 0

theorem isCritical_iff (p : Param K) : IsCritical p ↔ p = ⟨0, 0⟩ := by
  constructor
  · intro hp
    have h1 := hp ⟨1, 0⟩
    have h2 := hp ⟨0, 1⟩
    simp only [dρ, mul_one, mul_zero, add_zero, zero_add] at h1 h2
    ext <;> simp [h1, h2]
  · rintro rfl h; simp [dρ]

/-- **不完全なファイバー = 臨界値のファイバー。**
ゲージで説明できない同一視は、ちょうど `ρ` の特異点（臨界点）の上で起こる。 -/
theorem not_completeAt_iff_mem_critical_image (c : K) :
    ¬ CompleteAt Kˣ ρ c ↔ ∃ p, IsCritical p ∧ ρ p = c := by
  rw [completeAt_iff, not_not]
  constructor
  · rintro rfl; exact ⟨⟨0, 0⟩, (isCritical_iff _).mpr rfl, by simp [ρ]⟩
  · rintro ⟨p, hp, rfl⟩
    rw [(isCritical_iff p).mp hp]; simp [ρ]

/-- 不完全なファイバーは 2 本の座標軸の和（原点で正規交叉）。 -/
theorem fiber_zero :
    ρ ⁻¹' {(0 : K)} = {p : Param K | p.q = 0} ∪ {p | p.k = 0} := by
  ext p; simp [ρ]

/-- 原点の固定化群は全体、軸上の他の点の固定化群は自明：固定化群が跳ぶ。 -/
theorem stabilizer_origin : stabilizer Kˣ (⟨0, 0⟩ : Param K) = ⊤ := by
  ext t; simp [mem_stabilizer_iff, Param.ext_iff]

theorem stabilizer_axis {a : K} (ha : a ≠ 0) : stabilizer Kˣ (⟨a, 0⟩ : Param K) = ⊥ := by
  ext t
  simp only [mem_stabilizer_iff, Param.ext_iff, smul_q, smul_k, mul_zero, and_true,
    Subgroup.mem_bot]
  constructor
  · intro h
    apply Units.ext
    simpa using mul_right_cancel₀ ha (h.trans (one_mul a).symm)
  · rintro rfl; simp

end QK

end GaugeCompleteness
