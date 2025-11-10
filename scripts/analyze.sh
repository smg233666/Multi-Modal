#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Each model path on a separate line, ensuring correct separation.
"""
model_names=(
    "Qwen3-VL-2B-Thinking"
    "Qwen3-VL-4B-Thinking"
    "Qwen3-VL-8B-Thinking"
)
"""

# We first look at one model for testing
model_names=(
    "Qwen3-VL-4B-Thinking"
)

base_save_path="./Data/Eval/${dataname}/${model_basename}"
generation_save_path="${base_save_path}/${dataname}-${model_basename}_${steering_strength}_${steering_vector_type}_eval_vote_num${vote_num}.json"

for model_name in ${model_names[@]}; do
    echo "Analyzing $model_name"
    python AnalyzeOneModel.py \
    --model_name $model_name \
    --generation_save_path "$generation_save_path" \
    # break
done


