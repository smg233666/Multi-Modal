#!/bin/bash

# Default parameter values
model_path=${1:-"Qwen/Qwen3-VL-2B-Thinking"}
vote_num=${2:-8}
dataname=${3:-"MathVTest"}
tensor_parallel_size=${4:-1}
steering_strength=${5:-0.0}
steering_vector_type=${6:-"correct"}

model_basename=$(basename "$model_path")
base_save_path="./Data/Eval/${dataname}/${model_basename}"
overall_trend_save_path="${base_save_path}/overall_trend_results_${steering_vector_type}_vote_num${vote_num}.json"
generation_save_path="${base_save_path}/${dataname}-${model_basename}_${steering_strength}_${steering_vector_type}_eval_vote_num${vote_num}.json"
data_path="./Data/Questions/${dataname}.jsonl"

# Determine steering directory name without using associative arrays
case "$model_basename" in
  "Qwen3-VL-2B-Thinking"|"Qwen3-VL-4B-Thinking"|"Qwen3-VL-8B-Thinking")
    mapped_name="$model_basename"
    ;;
  *)
    mapped_name="$model_basename"
    ;;
esac

steering_vector_path="./Assets/MathV/${mapped_name}/mean_steering_vectors_${steering_vector_type}.npy"


echo "Running generator.py with following parameters:"
echo "Data path: $data_path"
echo "Model path: $model_path"
echo "Generation save path: $generation_save_path"
echo "Vote num: $vote_num"
echo "Dataset name: $dataname"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Steering vector path: $steering_vector_path"
echo "Steering strength: $steering_strength"
echo "Images root: ./Data/Images/MathV"
#echo "Images root: ./Data/Images/${dataname}"

python generate_and_evaluate_v.py \
  --dataname "$dataname" \
  --data_path "$data_path" \
  --model_path "$model_path" \
  --vote_num "$vote_num" \
  --tensor_parallel_size "$tensor_parallel_size" \
  --base_save_path "$base_save_path" \
  --generation_save_path "$generation_save_path" \
  --overall_trend_save_path "$overall_trend_save_path" \
  --steering_vector_path "$steering_vector_path" \
  --steering_strength "$steering_strength" \
  --images_root "./Data/Images/${dataname}" \
  --outputs_dir "./Data/Eval/${dataname}/Outputs" \
  --run_name "${model_basename}_${dataname}_multimodal"
