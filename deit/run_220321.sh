
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



# A_dist_4 dist cls0.5 g2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8


# B_dist_0 dist cls0.5 g2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12360 main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 1



# B_dist_0 dist cls0.5 g2
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12360 main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 0






# D
# D_dist_0 dist3 sum g2 r8
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12390 main.py \
    --model deit_small_patch16_224 --batch-size 256 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 \
    --fish_cls_token_type sum --fish_cls_token_pos -1 \
    --fish_global_proj_type mix --fish_layer_limit 8

# D_dist_1 dist3 sum g2m r8
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12391 main.py \
    --model deit_small_patch16_224 --batch-size 1 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 \
    --fish_cls_token_type sum --fish_cls_token_pos -1 \
    --fish_global_proj_type mix --fish_layer_limit 8

# D_dist_2 distq4 pos0.5 g2m r8
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12391 main.py \
    --model deit_small_patch16_224 --batch-size 1 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 4 --fish_mask_type distq \
    --fish_non_linear 0 --fish_non_linear_bias 0 \
    --fish_cls_token_type sum --fish_cls_token_pos 0.5 \
    --fish_global_proj_type mix --fish_layer_limit 8










# METRICS TEST ONLY
# rm -rf /host/ubuntu/vision/fishpp_metrics
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12380 main.py \
    --model deit_small_patch16_224 --batch-size 32 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_metrics \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 0 --metrics_test 1 --wandb 0 --eval


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12380 main.py \
    --model deit_small_patch16_224 --batch-size 32 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_metrics \
    --fishpp 1 --fish_global_heads 3 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 0 --metrics_test 1 --wandb 0 --eval


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12380 main.py \
    --model deit_small_patch16_224 --batch-size 32 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_metrics \
    --fishpp 1 --fish_global_heads 2 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 1 --metrics_test 1 --wandb 0 --eval


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12380 main.py \
    --model deit_small_patch16_224 --batch-size 32 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_metrics \
    --fishpp 1 --fish_global_heads 3 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 --fish_global_full_proj 1 \
    --fish_cls_token_pos 0.5 --fish_cls_token_proj 0 --fish_layer_limit 8 \
    --fish_global_full_mix 1 --metrics_test 1 --wandb 0 --eval


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12380 main.py \
    --model deit_small_patch16_224 --batch-size 32 --accumulation_steps 4 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_metrics \
    --fishpp 0 \
    --metrics_test 1 --wandb 0 --eval



