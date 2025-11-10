#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_paths=(
  "Qwen/Qwen3-VL-32B-Thinking"
)

data_path="Data/Questions/MathVMini.jsonl"
images_root="Data/Images/MathV"
save_path="Data/Representation/MathVMini"

for model_path in "${model_paths[@]}"; do
  echo "Processing model: ${model_path}"
  python embed.py \
    --reasoning True \
    --model_path "${model_path}" \
    --data_path "${data_path}" \
    --images_root "${images_root}" \
    --save_path "${save_path}"
done
