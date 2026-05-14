#!/bin/sh
set -eu

DATASET_PATH=${DATASET_PATH:-/data2/caiguoqing/Datasets/UniversalFakeDetect_full}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-29582}

CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON}" -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port ${PORT} \
  main.py \
  --method dream_cs \
  --model_variant clip_adapter \
  --dataset UniversalFakeDetect \
  --clip_path "${CLIP_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --train_selected_subsets car cat chair horse \
  --test_selected_subsets crn cyclegan biggan deepfake imle progan san stylegan \
  --batchsize 8 \
  --evalbatchsize 32 \
  --lr 0.00005 \
  --epoch 2 \
  --condition True \
  --gate True \
  --smooth True \
  --dream_num_experts 3 \
  --dream_rank 8 \
  --dream_fast_mode single_bank \
  --dream_fast_readout delta_from_prompt \
  --dream_fast_prompt_delta_norm True \
  --dream_fast_prompt_delta_scale_init 0.1 \
  --dream_fast_deep_bank True \
  --dream_fast_deep_residual True \
  --dream_residual_scale_init 1.0 \
  --dream_apply_init_bias -3.0 \
  --dream_warmup_epochs 1 \
  --dream_router_start_epoch 1 \
  --dream_route_margin 0.01 \
  --dream_num_train_views 1 \
  --dream_expert_forward_chunk 0 \
  --dream_save_pred_csv False \
  --dream_log_router True \
  --eval_interval 1 \
  --eval_max_batches_per_domain -1 \
  --num_workers 8 \
  --amp True \
  --amp_dtype bf16 \
  --loss_dream_clean 1.0 \
  --loss_dream_anchor 1.0 \
  --loss_dream_expert 0.3 \
  --loss_dream_specialize 0.3 \
  --loss_dream_anchor_rob 0.2 \
  --loss_dream_rob 0.5 \
  --loss_dream_inv 0.05 \
  --loss_dream_route 0.05 \
  --loss_dream_apply 0.02 \
  --loss_dream_clean_safe 0.5 \
  --loss_dream_res 0.01 \
  --loss_dream_div 0.01 \
  --model_name dream_cs_fast_single_bank_debug_seed100 \
  --seed 100 \
  --print_freq 50
