# pre_train
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 trainer/trainer_pretrain.py --use_wandb \
  --batch_size 16 \
  --epochs 6 \
  --total_batch_size_tokens 262144 \
  --num_hidden_layers 10
# python trainer/trainer_pretrain.py --use_wandb \
#   --batch_size  \
#   --epochs 6 \
#   --use_moe 1 \
#   --from_weight none \
#   --from_resume 0
# eval pre_train
# python eval.py --load_from ./out/pretrain_512.pth