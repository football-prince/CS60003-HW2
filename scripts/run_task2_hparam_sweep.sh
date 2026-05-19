#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Mac Apple Silicon defaults. Override them from the command line if needed:
# MODEL=resnet34 bash scripts/run_task2_hparam_sweep.sh
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

MODEL="${MODEL:-resnet18}"
DATA_DIR="${DATA_DIR:-./data}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
BACKBONE_SCALE="${BACKBONE_SCALE:-0.1}"
OVERWRITE="${OVERWRITE:-false}"
DOWNLOAD="${DOWNLOAD:-false}"

# Add more experiments here in the format:
# "epochs classifier_lr batch_size"
#
# classifier_lr is the lr in [epoch, lr, bs].
# backbone_lr is computed as classifier_lr * BACKBONE_SCALE.
CONFIGS=(
  # Baseline: moderate lr, standard setting
  "30 1e-3 32"

  # Learning rate analysis
  "30 1e-4 32"
  "30 5e-4 32"
  "30 2e-3 32"

  # Epoch analysis
  "10 1e-3 32"
  "50 1e-3 32"

  # Batch size analysis
  "30 1e-3 16"
)

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "$DOWNLOAD" == "true" ]]; then
  EXTRA_ARGS+=(--download)
fi

for CFG in "${CONFIGS[@]}"; do
  read -r EPOCHS LR BATCH_SIZE <<< "$CFG"
  BACKBONE_LR="$(
    python -c "print('{:.6g}'.format(float('$LR') * float('$BACKBONE_SCALE')))"
  )"

  echo "============================================================"
  echo "Task 2 sweep: model=${MODEL}, epochs=${EPOCHS}, classifier_lr=${LR}, backbone_lr=${BACKBONE_LR}, batch_size=${BATCH_SIZE}"
  echo "============================================================"

  python scripts/train_task2_hparam.py \
    --model "$MODEL" \
    --data_dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --backbone_lr "$BACKBONE_LR" \
    --classifier_lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --optimizer "$OPTIMIZER" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done

echo "All Task 2 sweep experiments finished."
