CKPT=${CKPT:-/path/to/dream_cs_checkpoint.pth}
DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect_full
if [ ! -d "${DEFAULT_DATASET_PATH}/test/dalle" ]; then
    DEFAULT_DATASET_PATH=/data2/caiguoqing/Datasets/UniversalFakeDetect
fi
DATASET_PATH=${DATASET_PATH:-${DEFAULT_DATASET_PATH}}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PYTHON=${PYTHON:-/data2/caiguoqing/.conda/envs/iapl/bin/python}
DATASET=${DATASET:-UniversalFakeDetect}
TEST_SELECTED_SUBSETS=${TEST_SELECTED_SUBSETS:-"crn cyclegan dalle biggan deepfake gaugan glide_50_27 glide_100_10 glide_100_27 guided imle ldm_100 ldm_200 ldm_200_cfg progan san seeingdark stargan stylegan"}

for DEG in none jpeg50 jpeg75 resize blur webp; do
    "${PYTHON}" main.py \
        --method dream_cs \
        --model_variant clip_adapter \
        --eval \
        --pretrained_model "$CKPT" \
        --clip_path "$CLIP_PATH" \
        --dataset_path "$DATASET_PATH" \
        --dataset "$DATASET" \
        --train_selected_subsets 'car' \
        --test_selected_subsets ${TEST_SELECTED_SUBSETS} \
        --batchsize 8 \
        --evalbatchsize 32 \
        --gate True \
        --condition True \
        --dream_eval_degradation "$DEG"
done
