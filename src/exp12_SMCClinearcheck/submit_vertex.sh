#!/usr/bin/env bash
# =====================================================================
# Vertex AI Custom Job submission for the SMCC/Markov falsification sweeps.
#
# WHY SHARDS: the models here are tiny (seq_len 11, ~1e6 params).  A single
# process leaves a GPU almost entirely idle -- wall clock is dominated by
# kernel-launch overhead, not FLOPs.  Running SHARDS concurrent processes on
# one device is therefore close to a linear speedup, and matters far more than
# picking a bigger accelerator.  For the same reason, do NOT reach for an A100:
# an L4 (or even a T4) is nearly as fast for this workload and far easier to
# get quota for in asia-northeast1.
#
# WHY GCS SYNC: Spot/preemptible workers lose local disk on restart.  Each
# shard appends to a local .jsonl and rsyncs it to GCS every SYNC_EVERY
# seconds; on startup it pulls any existing .jsonl back down, so the resume
# logic in exp2 picks up exactly where the preemption hit.
# =====================================================================
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
REGION=${REGION:-asia-northeast1}
BUCKET=${BUCKET:?set BUCKET (no gs:// prefix)}
REPO=${REPO:-ml-experiments}
IMAGE=${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/smcc-falsify:latest}
STAGE=${STAGE:-iso}                 # fast | iso | full | exp1
SHARDS=${SHARDS:-12}
MACHINE=${MACHINE:-g2-standard-8}   # L4; use n1-standard-8 + NVIDIA_TESLA_T4 if no L4 quota
ACCEL=${ACCEL:-NVIDIA_L4}
JOB="smcc-${STAGE}-$(date +%m%d-%H%M)"

case "$STAGE" in
  fast) PRESET=gpu_fast ;;
  iso)  PRESET=gpu_iso ;;
  full) PRESET=gpu_full ;;
  exp1) PRESET=gpu_full ;;
  *) echo "unknown STAGE $STAGE"; exit 1 ;;
esac

# ---------------------------------------------------------------- build
cat > Dockerfile <<'EOF'
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest
WORKDIR /app
COPY mult_common.py exp1_degree_probe.py exp2_multiplicity_sweep.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
EOF

cat > entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PRESET=${PRESET:?}; SHARDS=${SHARDS:-12}; GCS=${GCS:?}; STAGE=${STAGE:-iso}
SYNC_EVERY=${SYNC_EVERY:-120}
LOCAL=/tmp/results; mkdir -p "$LOCAL"

# pull any prior partial results so --jsonl resume can take effect
gsutil -m rsync -r "$GCS" "$LOCAL" 2>/dev/null || true

( while true; do sleep "$SYNC_EVERY"; gsutil -m rsync -r "$LOCAL" "$GCS" >/dev/null 2>&1 || true; done ) &
SYNCER=$!
trap 'kill $SYNCER 2>/dev/null || true; gsutil -m rsync -r "$LOCAL" "$GCS" || true' EXIT

if [ "$STAGE" = "exp1" ]; then
  python3 exp1_degree_probe.py --preset "$PRESET" --device cuda --stages 012 \
    --m_list 1 2 3 4 5 6 8 10 12 16 --L_list 4 --d_list 256 --seeds 2 \
    --jsonl "$LOCAL/exp1.jsonl" --out "$LOCAL/exp1.json"
  exit 0
fi

pids=()
for k in $(seq 0 $((SHARDS-1))); do
  python3 exp2_multiplicity_sweep.py --preset "$PRESET" --device cuda \
    --metric degmatch --shard "$k/$SHARDS" --light_save \
    --jsonl "$LOCAL/exp2_${STAGE}_s${k}.jsonl" \
    --out "$LOCAL/exp2_${STAGE}_s${k}.json" > "$LOCAL/log_s${k}.txt" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

# merge shards, then analyse under all three criteria
cat "$LOCAL"/exp2_${STAGE}_s*.jsonl > "$LOCAL/exp2_${STAGE}_merged.jsonl"
for M in degmatch r2_ood_matched r2_ood; do
  python3 exp2_multiplicity_sweep.py --analyze "$LOCAL/exp2_${STAGE}_merged.jsonl" \
    --metric "$M" > "$LOCAL/analysis_${STAGE}_${M}.txt" 2>&1 || true
done
exit $fail
EOF

gcloud builds submit --project "$PROJECT" --tag "$IMAGE" .

# ---------------------------------------------------------------- submit
cat > job.yaml <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE}
      acceleratorType: ${ACCEL}
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskSizeGb: 100
    containerSpec:
      imageUri: ${IMAGE}
      env:
        - name: PRESET
          value: "${PRESET}"
        - name: STAGE
          value: "${STAGE}"
        - name: SHARDS
          value: "${SHARDS}"
        - name: GCS
          value: "gs://${BUCKET}/smcc-falsify/${STAGE}"
EOF

gcloud ai custom-jobs create \
  --project "$PROJECT" --region "$REGION" \
  --display-name "$JOB" --config job.yaml

echo
echo "submitted: $JOB"
echo "results  : gs://${BUCKET}/smcc-falsify/${STAGE}/"
echo "tail     : gcloud ai custom-jobs stream-logs --region $REGION <JOB_ID>"
