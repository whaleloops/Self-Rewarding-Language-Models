#!/bin/sh
#
# baseline model 70371

conda activate mixtral_env
export HF_DATASETS_CACHE="/data/data_user_alpha/public_data"
export CUDA_DEVICE_ORDER="PCI_BUS_ID" 
export CUDA_HOME="/usr/local/cuda-12.1"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-12.1/bin:$PATH"
sleep 1

cd /home/zhichaoyang/Self-Rewarding-Language-Models
export TOKENIZER_MODEL=/data/data_user_alpha/public_models/Mistral/Mistral-7B-Instruct-v0.2
export BASE_MODEL=/data/data_user_alpha/public_models/Mistral/Mistral-7B-Instruct-v0.2
export PYTHONPATH=src
export CODE=/home/zhichaoyang/Self-Rewarding-Language-Models/scripts
export MODEL_NAME=M0
export DATA_DIR=/data/data_user_alpha/public_data/self_reward



# Train the instruction following and evaluation skills
CUDA_VISIBLE_DEVICES=3 python $CODE/00_sft.py -d $DATA_DIR/$MODEL_NAME/train/ift_eft.jsonl -b $TOKENIZER_MODEL -m $BASE_MODEL -o $DATA_DIR/$MODEL_NAME/models/sft

# Generate responses for the prompts
CUDA_VISIBLE_DEVICES=3 python $CODE/02_gen_responses.py $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/prompts_debug.jsonl $DATA_DIR/$MODEL_NAME/generated/responses.jsonl

# Generate responses for the prompts
CUDA_VISIBLE_DEVICES=2 python $CODE/03_gen_scores.py $TOKENIZER_MODEL $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/responses.jsonl $DATA_DIR/$MODEL_NAME/generated/scores.jsonl

# Create DPO data
CUDA_VISIBLE_DEVICES=2 python $CODE/04_gen_preferences.py $DATA_DIR/$MODEL_NAME/generated/scores.jsonl $DATA_DIR/$MODEL_NAME/generated/preferences.jsonl

# Train DPO model
CUDA_VISIBLE_DEVICES=2 python $CODE/05_dpo.py $TOKENIZER_MODEL $DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint $DATA_DIR/$MODEL_NAME/generated/preferences.jsonl $DATA_DIR/$MODEL_NAME/models/dpo/
