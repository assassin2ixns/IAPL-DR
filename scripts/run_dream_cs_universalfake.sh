DATASET_PATH=${DATASET_PATH:-/data2/caiguoqing/Datasets/UniversalFakeDetect}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
TRAIN_SELECTED_SUBSETS=${TRAIN_SELECTED_SUBSETS:-"car cat chair horse"}

# This local UniversalFakeDetect path points test/ to CNN_synth_testset.
# It does not contain dalle/glide/guided/ldm domains.
TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan biggan deepfake gaugan imle progan san seeingdark stargan stylegan stylegan2 whichfaceisreal"}

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

"${PYTHON}" -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29582 \
    main.py \
    --method dream_cs \
    --model_variant clip_adapter \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "${CLIP_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --dataset UniversalFakeDetect \
    --train_selected_subsets ${TRAIN_SELECTED_SUBSETS} \
    --test_selected_subsets ${TEST_SELECTED_SUBSETS} \
    --lr 0.00005 \
    --model_name dream_cs_standalone_ufd_progan_v03 \
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True \
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
    --loss_dream_div 0.005

# If training OOMs: lower --batchsize from 8 to 4, set --dream_num_train_views 1,
# set --dream_expert_forward_chunk 8 or 4, optionally debug with
# --dream_num_experts 2, and lower --evalbatchsize to 16 if evaluation OOMs.
# DREAM-CS Standalone does not pass --dream_anchor_ckpt and does not freeze anchor.
