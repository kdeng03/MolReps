#!/bin/bash

#SBATCH --job-name=grpo
#SBATCH --output=log/grpo-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

nvidia-smi

# source .env

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm

# module load cudatoolkit
# module load brics/openmpi/4.1.7
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12


task_name=mol_rep_conversion
# task_name=mol_rep_ocr

# base_model_name=qwen3_4b_i
base_model_name=qwen3_vl_4b_i

# parent_training_method=base
parent_training_method=sft_lora

training_method=grpo_lora

mkdir -p logs/grpo_trl/v1.1/$task_name

### Single GPU training
PYTHONPATH="$(pwd)" \
    accelerate launch \
    scripts/grpo_trl.py \
    --task_name "$task_name" \
    --base_model_name "$base_model_name" \
    --parent_training_method "$parent_training_method" \
    --training_method "$training_method" | tee logs/grpo_trl/v1.1/$task_name/${base_model_name}_${training_method}_from_${parent_training_method}.log
