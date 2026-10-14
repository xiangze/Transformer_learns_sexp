#!/usr/bin/env bash
# Staged GPU sweeps.  Same code path as the CPU runs -- only --preset changes.
# Every stage appends to a .jsonl and resumes if interrupted, so re-running the
# same command after a preemption picks up where it stopped.
set -euo pipefail

OUT=${OUT:-./results}
DEV=${DEV:-cuda}
mkdir -p "$OUT"

echo "== stage A: instrument re-calibration at GPU training budget =="
# Exp1 S2 with 30k steps.  The point is resolution: the degree probe's ceiling
# is set by the model's own fit noise, so a better-trained model resolves higher
# m.  On CPU (1k steps, noise ~3.8%) the probe saturated at 6; this establishes
# where it saturates when noise is an order of magnitude smaller.
python3 exp1_degree_probe.py \
  --preset gpu_full --device "$DEV" --stages 012 \
  --m_list 1 2 3 4 5 6 8 10 12 16 --L_list 4 --d_list 256 --seeds 2 \
  --jsonl "$OUT/exp1_gpu.jsonl" --out "$OUT/exp1_gpu.json"

echo "== stage B: gpu_fast -- coarse grid, 10k steps, 1 seed (~1.5 h A100) =="
python3 exp2_multiplicity_sweep.py \
  --preset gpu_fast --device "$DEV" --metric degmatch \
  --jsonl "$OUT/exp2_fast.jsonl" --out "$OUT/exp2_fast.json"

echo "== stage C: gpu_iso -- iso-parameter control, L*d^2 held constant (~5 h) =="
# This is the decisive control: it separates "depth helps" from "capacity helps".
python3 exp2_multiplicity_sweep.py \
  --preset gpu_iso --device "$DEV" --metric degmatch \
  --jsonl "$OUT/exp2_iso.jsonl" --out "$OUT/exp2_iso.json"

echo "== stage D: gpu_full -- full grid, 3 seeds (~30 h A100) =="
python3 exp2_multiplicity_sweep.py \
  --preset gpu_full --device "$DEV" --metric degmatch \
  --jsonl "$OUT/exp2_full.jsonl" --out "$OUT/exp2_full.json"

echo "== re-analysis under all three criteria =="
for f in exp2_fast exp2_iso exp2_full; do
  for M in degmatch r2_ood_matched r2_ood; do
    echo "--- $f / $M ---"
    python3 exp2_multiplicity_sweep.py --analyze "$OUT/$f.jsonl" --metric "$M" \
      | tail -20
  done
done
