python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29585 \
    main.py \
    --method dream_cs \
    --model_variant clip_adapter \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "/path/to/ViT-L-14.pt" \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'SDv14' \
    --test_selected_subsets 'ADM' 'BigGAN' 'glide' 'Midjourney' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5' 'VQDM' 'wukong' \
    --lr 0.005 \
    --model_name tta_dream_cs_genimage \
    --dataset GenImage \
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model /path/to/dream_cs_genimage_checkpoint.pth \
    --eval \
    --smooth True \
    --tta True \
    --tta_steps 2 \
    --ois True \
    --dream_tta_safe True \
    --dream_tta_agg mean \
    --dream_tta_disagreement_fallback True \
    --dream_tta_disagreement_thresh 0.05 \
    --num_workers 8

# Safe TTA freezes router, expert residuals, fc_binary, conditional_ctx, and adapters;
# only prompt_learner.ctx is optimized for each test image.
