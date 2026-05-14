#!/bin/sh
set -eu

DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect_full
if [ ! -d "${DEFAULT_DATASET_PATH}/test/dalle" ]; then
    DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect
fi
DATASET_PATH=${DATASET_PATH:-${DEFAULT_DATASET_PATH}}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
BATCHSIZE=${BATCHSIZE:-8}
EVALBATCHSIZE=${EVALBATCHSIZE:-32}
EPOCH=${EPOCH:-5}
LR=${LR:-0.00005}
PRINT_FREQ=${PRINT_FREQ:-50}
TRAIN_SELECTED_SUBSETS=${TRAIN_SELECTED_SUBSETS:-"car cat chair horse"}

if [ -d "${DATASET_PATH}/test/dalle" ]; then
    TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan dalle biggan deepfake gaugan glide_50_27 glide_100_10 glide_100_27 guided imle ldm_100 ldm_200 ldm_200_cfg progan san seeingdark stargan stylegan"}
else
    # Legacy local CNN_synth_testset subset.
    TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan biggan deepfake gaugan imle progan san seeingdark stargan stylegan stylegan2 whichfaceisreal"}
fi

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
        --dream_residual_scale_init 0.001 \
        --dream_apply_init_bias -3.0 \
        --dream_warmup_epochs 1 \
        --dream_router_start_epoch 1 \
        --dream_num_train_views 2 \
        --dream_expert_forward_chunk 16 \
        --dream_log_router True \
        --dream_save_pred_csv True \
        --loss_dream_clean 1.0 \
        --loss_dream_anchor 1.0 \
        --loss_dream_expert 0.2 \
        --loss_dream_specialize 0.2 \
        --loss_dream_anchor_rob 0.2 \
        --loss_dream_rob 0.5 \
        --loss_dream_inv 0.05 \
        --loss_dream_route 0.1 \
        --loss_dream_apply 0.05 \
        --loss_dream_clean_safe 0.5 \
        --loss_dream_res 0.01 \
        --loss_dream_div 0.005 \
        --model_name "${NAME}" \
        --seed ${SEED} \
        --print_freq ${PRINT_FREQ}
}

run_one 6 100 29582 dream_cs_standalone_ufd_progan_v03_seed100 > dream_cs_gpu6_seed100.log 2>&1 &
PID6=$!

run_one 7 101 29583 dream_cs_standalone_ufd_progan_v03_seed101 > dream_cs_gpu7_seed101.log 2>&1 &
PID7=$!

echo "Started DREAM-CS Standalone jobs:"
echo "  GPU 6 seed 100 pid=${PID6} log=dream_cs_gpu6_seed100.log"
echo "  GPU 7 seed 101 pid=${PID7} log=dream_cs_gpu7_seed101.log"
echo "Monitor with:"
echo "  tail -f dream_cs_gpu6_seed100.log"
echo "  tail -f dream_cs_gpu7_seed101.log"

wait ${PID6}
wait ${PID7}
