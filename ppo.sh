#!/usr/bin/env bash

HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node=6 trainer/trainer_ppo.py \
  --use_wandb \
  --from_resume 1 \
  --save_dir out \
  --save_weight ppo_actor \
  --epochs 2 \
  --batch_size 2 \
  --learning_rate 8e-8 \
  --critic_learning_rate 8e-8 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --reasoning 0 \
  --data_path dataset/rlaif-mini.jsonl \
  --reward_model_path internlm/internlm2-1_8b-reward
