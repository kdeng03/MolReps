#!/bin/bash

#SBATCH --job-name=pass_at_k
#SBATCH --output=logs/pass_at_k-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

nvidia-smi

# source .env

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm

# === Configuration ===
# Model name (defined in configs/model_inference_schema.yaml)
model_name=qwen3_vl_4b_i_sft_lora_ocr_conversion

# Dataset configuration
task_name=mol-rep-conversion
dataset_name=kdeng03/${task_name}-v1.1
dataset_split=test

# Evaluation configuration
max_k=16
max_samples=-1
temperature=1.0
top_p=1.0
max_new_tokens=2048

# Output configuration
output_file=results/pass_at_k/${task_name}/v1.1/${dataset_split}/${model_name}_pass_at_${max_k}.json

# Data field configuration (adjust based on your dataset)
prompt_key=prompt
image_key=input_rep

# === Create log and result directories ===
mkdir -p logs/pass_at_k/${task_name}/v1.1/${dataset_split}
mkdir -p results/pass_at_k/${task_name}/v1.1/${dataset_split}

### 运行 Pass@k 评估
PYTHONPATH="$(pwd)" \
    python \
    scripts/eval_pass_at_k.py \
    --engine_name "$model_name" \
    --dataset_name "$dataset_name" \
    --dataset_split "$dataset_split" \
    --max_k "$max_k" \
    --max_samples "$max_samples" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --max_new_tokens "$max_new_tokens" \
    --output_file "$output_file" \
    --prompt_key "$prompt_key" \
    --image_key "$image_key" 2>&1 | tee logs/pass_at_k/${task_name}/v1.1/${dataset_split}/${model_name}_pass_at_${max_k}.log
