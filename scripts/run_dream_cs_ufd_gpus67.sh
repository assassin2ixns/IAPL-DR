#!/bin/sh
set -eu

# DREAM-CS-Fast two-seed runner on GPU 6 and GPU 7.
# Default FAST_MODE=single_bank is the speed-first setting for debugging/trial runs.
# Set FAST_MODE=bank_plus_anchor for the no-regret-safe main candidate.
# This standalone setting does not load an IAPL checkpoint.

DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect_full
if [ ! -d "${DEFAULT_DATASET_PATH}/test/dalle" ]; then
    DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect
fi

DATASET_PATH=${DATASET_PATH:-${DEFAULT_DATASET_PATH}}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}

FAST_MODE=${FAST_MODE:-single_bank}
BATCHSIZE=${BATCHSIZE:-8}
EVALBATCHSIZE=${EVALBATCHSIZE:-32}
EPOCH=${EPOCH:-2}
LR=${LR:-0.00005}
PRINT_FREQ=${PRINT_FREQ:-50}
NUM_WORKERS=${NUM_WORKERS:-8}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
EVAL_MAX_BATCHES_PER_DOMAIN=${EVAL_MAX_BATCHES_PER_DOMAIN:--1}
FULL_EVAL=${FULL_EVAL:-0}

GPU6=${GPU6:-6}
GPU7=${GPU7:-7}
SEED6=${SEED6:-100}
SEED7=${SEED7:-101}
PORT6=${PORT6:-29582}
PORT7=${PORT7:-29583}

TRAIN_SELECTED_SUBSETS=${TRAIN_SELECTED_SUBSETS:-"car cat chair horse"}

if [ "${FULL_EVAL}" = "1" ] && [ -d "${DATASET_PATH}/test/dalle" ]; then
    TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan dalle biggan deepfake gaugan glide_50_27 glide_100_10 glide_100_27 guided imle ldm_100 ldm_200 ldm_200_cfg progan san seeingdark stargan stylegan"}
else
    TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan biggan deepfake imle progan san stylegan"}
fi

case "${FAST_MODE}" in
    off|bank_plus_anchor|single_bank) ;;
    *)
        echo "FAST_MODE must be one of: off, bank_plus_anchor, single_bank" >&2
        exit 1
        ;;
esac

for subset in ${TRAIN_SELECTED_SUBSETS}; do
    if [ ! -d "${DATASET_PATH}/train/${subset}" ]; then
        echo "Missing train subset: ${DATASET_PATH}/train/${subset}" >&2
        exit 1
    fi
done

for subset in ${TEST_SELECTED_SUBSETS}; do
    if [ ! -d "${DATASET_PATH}/test/${subset}" ]; then
        echo "Missing test subset: ${DATASET_PATH}/test/${subset}" >&2
        echo "Available test subsets:" >&2
        find -L "${DATASET_PATH}/test" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort >&2
        exit 1
    fi
done

run_one() {
    GPU_ID=$1
    SEED=$2
    PORT=$3
    NAME=$4

    CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON}" -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port ${PORT} \
        main.py \
        --method dream_cs \
        --model_variant clip_adapter \
        --dataset UniversalFakeDetect \
        --clip_path "${CLIP_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --train_selected_subsets ${TRAIN_SELECTED_SUBSETS} \
        --test_selected_subsets ${TEST_SELECTED_SUBSETS} \
        --batchsize ${BATCHSIZE} \
        --evalbatchsize ${EVALBATCHSIZE} \
        --lr ${LR} \
        --epoch ${EPOCH} \
        --lr_drop 10 \
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
        --eval_interval ${EVAL_INTERVAL} \
        --eval_max_batches_per_domain ${EVAL_MAX_BATCHES_PER_DOMAIN} \
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
        --model_name "${NAME}" \
        --seed ${SEED} \
        --print_freq ${PRINT_FREQ}
}

NAME6=dream_cs_fast_${FAST_MODE}_ufd_seed${SEED6}
NAME7=dream_cs_fast_${FAST_MODE}_ufd_seed${SEED7}
LOG6=dream_cs_fast_${FAST_MODE}_gpu${GPU6}_seed${SEED6}.log
LOG7=dream_cs_fast_${FAST_MODE}_gpu${GPU7}_seed${SEED7}.log

run_one ${GPU6} ${SEED6} ${PORT6} ${NAME6} > "${LOG6}" 2>&1 &
PID6=$!

run_one ${GPU7} ${SEED7} ${PORT7} ${NAME7} > "${LOG7}" 2>&1 &
PID7=$!

echo "Started DREAM-CS-Fast jobs:"
echo "  GPU ${GPU6} seed ${SEED6} pid=${PID6} log=${LOG6} output=outputs/${NAME6}"
echo "  GPU ${GPU7} seed ${SEED7} pid=${PID7} log=${LOG7} output=outputs/${NAME7}"
echo "Config: FAST_MODE=${FAST_MODE} EPOCH=${EPOCH} FULL_EVAL=${FULL_EVAL} EVAL_MAX_BATCHES_PER_DOMAIN=${EVAL_MAX_BATCHES_PER_DOMAIN}"
echo "Monitor with:"
echo "  tail -f ${LOG6}"
echo "  tail -f ${LOG7}"

wait ${PID6}
wait ${PID7}
