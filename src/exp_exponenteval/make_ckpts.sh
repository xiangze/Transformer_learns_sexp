#!/bin/bash
# Build every checkpoint the eval probe needs, at MATCHED step counts, over
# several training seeds.
#
# Matched steps matter: the null expectations for rho and sep_inv scale with
# each model's own Jacobian error, so controls trained for fewer steps are not
# comparable.  Several seeds matter because a probe CI covers the sampling of
# probe points within one model, not whether another training run lands in the
# same place.
set -e
S=${STEPS:-50000}
B=${BSZ:-512}
LR=${LR:-1e-3}
EV=${SAVE_EVERY:-5000}
SEEDS=${SEEDS:-"0 1 2"}
PATHS=${PATHS:-full,noA,noV,noVnum,noVprog,noMLP,frozen}
COMMON="--steps $S --bsz $B --lr $LR --save-every $EV"

probe () {   # probe <ckpt> <json> [extra args...]
  python probe_mlp.py --ckpt "$1" --n-u 64 --n-v 16 --n-pairs 64 \
      --paths "$PATHS" --json "$2" "${@:3}" 2>&1 | tee "${2%.json}.txt"
}

for s in $SEEDS; do
  echo "=============== seed $s ==============="
  # --- main model + checkpoint series
  python train_cont.py --mode eval $COMMON --seed $s \
      --out ckpt_eval50k_s$s.pt   2>&1 | tee log_eval50k_s$s.txt
  probe ckpt_eval50k_s$s.pt res_eval50k_s$s.json --v-dist gauss

  # --- code-blind floor, same steps
  python train_cont.py --mode ignore_code $COMMON --seed $s \
      --out ckpt_ignore50k_s$s.pt 2>&1 | tee log_ignore50k_s$s.txt
  probe ckpt_ignore50k_s$s.pt res_ignore50k_s$s.json --v-dist gauss

  # --- random-init floor (--steps 0 short-circuits training)
  python train_cont.py --mode eval --steps 0 --seed $s \
      --out ckpt_rand50k_s$s.pt   2>&1 | tee log_rand50k_s$s.txt
  probe ckpt_rand50k_s$s.pt res_rand50k_s$s.json --v-dist gauss

  # --- linear-only value embedding: Fourier feature frequencies enter the
  #     characteristic-polynomial statistics directly, so the conclusion has to
  #     survive without them
  python train_cont.py --mode eval $COMMON --seed $s --numemb linear \
      --out ckpt_evallin50k_s$s.pt 2>&1 | tee log_evallin50k_s$s.txt
  probe ckpt_evallin50k_s$s.pt res_evallin50k_s$s.json --v-dist gauss

  # --- TRUE lookup control: 156 programs instead of 47988, so a table is
  #     cheaper than learning the composition.  MAX_DEPTH>=2 or rho has no
  #     composable pairs.  Its own language -> its own dir, and it must be
  #     probed with the same SEXP_* variables.
  SEXP_N_ANGLE=4 SEXP_MAX_DEPTH=2 python train_cont.py --mode grid $COMMON \
      --seed $s --out lookup/ckpt_lookup50k_s$s.pt 2>&1 | tee log_lookup50k_s$s.txt
  SEXP_N_ANGLE=4 SEXP_MAX_DEPTH=2 probe lookup/ckpt_lookup50k_s$s.pt \
      res_lookup50k_s$s.json --v-dist grid
done

echo "=============== across-seed aggregation ==============="
for fam in eval ignore rand evallin lookup; do
  ls res_${fam}50k_s*.json >/dev/null 2>&1 && \
    { echo; python agg_seeds.py res_${fam}50k_s*.json; }
done
