python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29583 \
    main.py \
    --method dream_cs \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "/path/to/ViT-L-14.pt" \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'SDv14' \
    --test_selected_subsets 'ADM' 'BigGAN' 'glide' 'Midjourney' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5' 'VQDM' 'wukong' \
    --lr 0.00005 \
    --model_name dream_cs_genimage_sd14 \
    --dataset GenImage \
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --smooth True \
    --dream_num_experts 3 \
    --dream_rank 8 \
    --dream_num_train_views 2 \
    --dream_expert_forward_chunk 16 \
    --loss_dream_clean 1.0 \
    --loss_dream_anchor 0.2 \
    --loss_dream_rob 0.5 \
    --loss_dream_inv 0.05 \
    --loss_dream_route 0.1 \
    --loss_dream_apply 0.05 \
    --loss_dream_clean_safe 0.5 \
    --loss_dream_res 0.01

# If training OOMs: lower --batchsize from 8 to 4, set --dream_num_train_views 1,
# set --dream_expert_forward_chunk 8 or 4, optionally debug with
# --dream_num_experts 2, and lower --evalbatchsize to 16 if evaluation OOMs.
