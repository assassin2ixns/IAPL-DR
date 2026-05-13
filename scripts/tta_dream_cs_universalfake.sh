python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29584 \
    main.py \
    --method dream_cs \
    --model_variant clip_adapter \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "/path/to/ViT-L-14.pt" \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'car' 'cat' 'chair' 'horse' \
    --test_selected_subsets 'crn' 'cyclegan' 'dalle' 'biggan' 'deepfake' 'gaugan' 'glide_50_27' 'glide_100_10' 'glide_100_27' 'guided' 'imle' 'ldm_100' 'ldm_200' 'ldm_200_cfg' 'progan' 'san' 'seeingdark' 'stargan' 'stylegan' \
    --lr 0.005 \
    --model_name tta_dream_cs_universalfake \
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model /path/to/dream_cs_universalfake_checkpoint.pth \
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
