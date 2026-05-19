#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Mac Apple Silicon friendly defaults.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

DATA_DIR="${DATA_DIR:-./data}"
DEVICE="${DEVICE:-mps}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-30}"
TASK2_CONFIGS="${TASK2_CONFIGS:-30:1e-3:32,30:1e-4:32,30:5e-4:32,30:2e-3:32,10:1e-3:32,50:1e-3:32,30:1e-3:16}"
INCLUDE_TRANSFORMERS="${INCLUDE_TRANSFORMERS:-true}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-8}"
TRANSFORMER_LR="${TRANSFORMER_LR:-1e-4}"
OVERWRITE="${OVERWRITE:-false}"

COMMON_ARGS=(
  --data_dir "$DATA_DIR"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --optimizer "$OPTIMIZER"
  --weight_decay "$WEIGHT_DECAY"
)

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

run_cmd() {
  echo
  echo "============================================================"
  echo "$*"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  "$@"
  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
}

run_task1() {
  local model="$1"
  run_cmd python scripts/train_task1_baseline.py \
    --model "$model" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --backbone_lr 1e-4 \
    --classifier_lr 1e-3 \
    --output_dir "outputs/task1_${model}" \
    "${COMMON_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

run_task2() {
  local model="$1"
  IFS=',' read -ra configs <<< "$TASK2_CONFIGS"
  for cfg in "${configs[@]}"; do
    IFS=':' read -r epochs lr bs <<< "$cfg"
    local backbone_lr
    backbone_lr="$(
      python -c "print('{:.6g}'.format(float('$lr') * 0.1))"
    )"
    run_cmd python scripts/train_task2_hparam.py \
      --model "$model" \
      --epochs "$epochs" \
      --batch_size "$bs" \
      --backbone_lr "$backbone_lr" \
      --classifier_lr "$lr" \
      "${COMMON_ARGS[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  done
}

run_task3() {
  local model="$1"
  run_cmd python scripts/train_task3_ablation.py \
    --model "$model" \
    --pretrained true \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --backbone_lr 1e-4 \
    --classifier_lr 1e-3 \
    "${COMMON_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

  run_cmd python scripts/train_task3_ablation.py \
    --model "$model" \
    --pretrained false \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --backbone_lr 1e-3 \
    --classifier_lr 1e-3 \
    "${COMMON_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

run_task4() {
  local model="$1"
  for attention in se cbam; do
    run_cmd python scripts/train_task4_attention.py \
      --model "$model" \
      --attention "$attention" \
      --pretrained true \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --backbone_lr 1e-4 \
      --classifier_lr 1e-3 \
      "${COMMON_ARGS[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  done
}

run_task4_transformers() {
  for model in vit_tiny swin_t; do
    run_cmd python scripts/train_task4_attention.py \
      --model "$model" \
      --pretrained true \
      --epochs "$EPOCHS" \
      --batch_size "$TRANSFORMER_BATCH_SIZE" \
      --lr "$TRANSFORMER_LR" \
      "${COMMON_ARGS[@]}" \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  done
}

echo "Preparing Flowers102 dataset..."
run_cmd python scripts/prepare_data.py --data_dir "$DATA_DIR"

for model in resnet18 resnet34; do
  echo
  echo "############################################################"
  echo "Running all ResNet experiments for ${model}"
  echo "############################################################"
  run_task1 "$model"
  run_task2 "$model"
  run_task3 "$model"
  run_task4 "$model"
done

if [[ "$INCLUDE_TRANSFORMERS" == "true" ]]; then
  echo
  echo "############################################################"
  echo "Running Task 4 Transformer experiments: ViT-Tiny and Swin-T"
  echo "############################################################"
  run_task4_transformers
fi

echo
echo "All experiments finished at: $(date '+%Y-%m-%d %H:%M:%S')"
