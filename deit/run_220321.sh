
# 
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --output_dir /host/ubuntu/vision/fishpp \
    --accumulation_steps 4




# A_dist_0 dist cls0.5
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0


# A_dist_1 dist cls0.5 g2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0


# A_hdist_0 cls
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_levels 3 --fish_mask_type hdist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos -1 --fish_cls_token_proj 1

