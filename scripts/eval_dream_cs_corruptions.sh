CKPT=${CKPT:-/path/to/dream_cs_checkpoint.pth}
DATASET_PATH=${DATASET_PATH:-/path/to/dataset}
CLIP_PATH=${CLIP_PATH:-/path/to/ViT-L-14.pt}
DATASET=${DATASET:-UniversalFakeDetect}

for DEG in none jpeg50 jpeg75 resize blur webp; do
    python main.py \
        --method dream_cs \
        --eval \
        --pretrained_model "$CKPT" \
        --clip_path "$CLIP_PATH" \
        --dataset_path "$DATASET_PATH" \
        --dataset "$DATASET" \
        --train_selected_subsets 'car' \
        --test_selected_subsets 'progan' \
        --batchsize 8 \
        --evalbatchsize 32 \
        --gate True \
        --condition True \
        --dream_eval_degradation "$DEG"
done
