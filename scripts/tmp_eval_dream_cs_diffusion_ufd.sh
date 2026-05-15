#!/bin/sh
set -eu

# Temporary eval script: evaluate the current DREAM-CS-Fast checkpoint on UFD diffusion subsets only.
# Results append to outputs/${MODEL_NAME}/eval_domain_metrics.jsonl and eval_mean_metrics.jsonl.

DATASET_PATH=${DATASET_PATH:-/data2/caiguoqing/Datasets/UniversalFakeDetect_full}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
GPUS=${GPUS:-6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29682}
RUN_LOG=${RUN_LOG:-dream_cs_fast_bank_plus_anchor_ddp_gpus67_seed100.log}

MODEL_NAME=${MODEL_NAME:-dream_cs_fast_bank_plus_anchor_ufd_ddp_gpus67_seed100}
CKPT=${CKPT:-outputs/${MODEL_NAME}/checkpoint_last.pth}
EVALBATCHSIZE=${EVALBATCHSIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
FAST_MODE=${FAST_MODE:-bank_plus_anchor}

DIFFUSION_SUBSETS=${DIFFUSION_SUBSETS:-"dalle glide_50_27 glide_100_10 glide_100_27 guided ldm_100 ldm_200 ldm_200_cfg"}

if [ ! -f "${CKPT}" ]; then
    echo "Missing checkpoint: ${CKPT}" >&2
    exit 1
fi

for subset in ${DIFFUSION_SUBSETS}; do
    if [ ! -d "${DATASET_PATH}/test/${subset}" ]; then
        echo "Missing diffusion test subset: ${DATASET_PATH}/test/${subset}" >&2
        echo "Available test subsets:" >&2
        find -L "${DATASET_PATH}/test" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort >&2
        exit 1
    fi
done

log_msg() {
    echo "$1"
    echo "$1" >> "${RUN_LOG}"
}

log_msg ""
log_msg "[DIFFUSION-EVAL] Evaluating DREAM-CS-Fast on UFD diffusion subsets:"
log_msg "[DIFFUSION-EVAL] checkpoint=${CKPT}"
log_msg "[DIFFUSION-EVAL] model_name=${MODEL_NAME}"
log_msg "[DIFFUSION-EVAL] subsets=${DIFFUSION_SUBSETS}"
log_msg "[DIFFUSION-EVAL] gpus=${GPUS} nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
log_msg "[DIFFUSION-EVAL] detailed metrics append to outputs/${MODEL_NAME}/eval_domain_metrics.jsonl"
log_msg "[DIFFUSION-EVAL] console output append to ${RUN_LOG}"

run_eval() {
CUDA_VISIBLE_DEVICES=${GPUS} "${PYTHON}" -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port ${MASTER_PORT} \
    main.py \
    --eval \
    --method dream_cs \
    --model_variant clip_adapter \
    --dataset UniversalFakeDetect \
    --clip_path "${CLIP_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --train_selected_subsets car cat chair horse \
    --test_selected_subsets ${DIFFUSION_SUBSETS} \
    --batchsize 8 \
    --evalbatchsize ${EVALBATCHSIZE} \
    --lr 0.00005 \
    --epoch 1 \
    --condition True \
    --gate True \
    --smooth True \
    --dream_num_experts 3 \
    --dream_rank 8 \
    --dream_fast_mode ${FAST_MODE} \
    --dream_fast_readout delta_from_prompt \
    --dream_fast_prompt_delta_norm True \
    --dream_fast_prompt_delta_scale_init 0.1 \
    --dream_fast_deep_bank True \
    --dream_fast_deep_residual True \
    --dream_residual_scale_init 1.0 \
    --dream_apply_init_bias -3.0 \
    --dream_warmup_epochs 1 \
    --dream_warmup_freeze_router True \
    --dream_warmup_fixed_apply 0.2 \
    --dream_router_start_epoch 1 \
    --dream_route_margin 0.01 \
    --dream_num_train_views 1 \
    --dream_balanced_degradation_views True \
    --dream_expert_forward_chunk 0 \
    --dream_save_pred_csv False \
    --dream_log_router True \
    --eval_max_batches_per_domain -1 \
    --num_workers ${NUM_WORKERS} \
    --amp True \
    --amp_dtype bf16 \
    --tf32 True \
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
    --model_name "${MODEL_NAME}" \
    --pretrained_model "${CKPT}" \
    --seed 100 \
    --print_freq 50
}

FIFO=$(mktemp -u "/tmp/dream_cs_diffusion_eval.XXXXXX")
mkfifo "${FIFO}"
tee -a "${RUN_LOG}" < "${FIFO}" &
TEE_PID=$!
set +e
run_eval > "${FIFO}" 2>&1
STATUS=$?
set -e
wait "${TEE_PID}" 2>/dev/null || true
rm -f "${FIFO}"
exit "${STATUS}"
