# Plugin/warmstart ablation only. This is not the main DREAM-CS Standalone setting.
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29586 \
    main.py \
    --method dream_cs \
    --model_variant clip_adapter \
    --batchsize 8 \
    --evalbatchsize 32 \
    --clip_path "/path/to/ViT-L-14.pt" \
    --dataset_path "/path/to/dataset" \
    --dataset UniversalFakeDetect \
    --train_selected_subsets 'car' 'cat' 'chair' 'horse' \
    --test_selected_subsets 'progan' \
    --lr 0.00005 \
    --model_name dream_cs_plugin_iapl_anchor_ufd \
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --smooth True \
    --dream_anchor_ckpt "/path/to/iapl_anchor_checkpoint.pth" \
    --dream_num_experts 3 \
    --dream_rank 8 \
    --dream_residual_scale_init 0.001 \
    --dream_apply_init_bias -3.0 \
    --dream_num_train_views 2 \
    --dream_expert_forward_chunk 16
