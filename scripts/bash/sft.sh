#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=logs/sft-%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

nvidia-smi

# source .env

source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm
# conda activate qwen3_5

# module load cuda
# module load brics/openmpi/4.1.7
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12

task_name=mol_rep_conversion
# task_name=mol_rep_ocr
# task_name=mol_rep_ocr_conversion

# model_name=qwen3_4b_i
model_name=qwen3_vl_4b_i

training_method=sft_lora

mkdir -p logs/sft/v1.1/$task_name

### Single GPU training
PYTHONPATH="$(pwd)" \
    accelerate launch \
    scripts/sft.py \
    --task_name "$task_name" \
    --model_name "$model_name" \
    --training_method "$training_method" | tee logs/sft/v1.1/$task_name/$model_name.log

# ### Distributed training with DeepSpeed
# PYTHONPATH="$(pwd)" \
#     accelerate launch \
#     --config_file configs/sft_zero3.yaml \
#     scripts/sft.py \
#     --task_name "$task_name" \
#     --model_name "$model_name" \
#     --training_method "$training_method"
