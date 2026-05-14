DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect_full
if [ ! -d "${DEFAULT_DATASET_PATH}/test/dalle" ]; then
    DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect
fi
DATASET_PATH=${DATASET_PATH:-${DEFAULT_DATASET_PATH}}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-/path/to/dream_cs_universalfake_checkpoint.pth}
TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan dalle biggan deepfake gaugan glide_50_27 glide_100_10 glide_100_27 guided imle ldm_100 ldm_200 ldm_200_cfg progan san seeingdark stargan stylegan"}

"${PYTHON}" -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29584 \
    main.py \
    --method dream_cs \
    --model_variant clip_adapter \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "${CLIP_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --train_selected_subsets 'car' 'cat' 'chair' 'horse' \
    --test_selected_subsets ${TEST_SELECTED_SUBSETS} \
    --lr 0.005 \
    --model_name tta_dream_cs_universalfake \
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model "${PRETRAINED_MODEL}" \
    --eval \
    --tta True \
    --tta_steps 2 \
    --ois True \
    --dream_tta_safe True \
    --dream_tta_agg mean \
    --dream_tta_disagreement_fallback True \
    --dream_tta_disagreement_thresh 0.05

# Safe TTA freezes router, expert residuals, fc_binary, conditional_ctx, and adapters;
# only prompt_learner.ctx is optimized for each test image.
