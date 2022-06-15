
export http_proxy=http://10.16.29.10:8080
export https_proxy=http://10.16.29.10:8080



data_path="/docker/code/data/imagenet2012"
output_path="/docker/code/data/vision/fishpp"
path="--data-path "$data_path" --output_dir "$output_path
gpus="6,7"
gpu_count=2
_cuda="-m torch.distributed.launch --nproc_per_node="$gpu_count" --use_env --master_port"
_model="--model deit_small_patch16_224"

# bs=256
# accum=4
bs=512
accum=1

# E_dist_0 dist3 sum g2 r8
CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12400 main.py \
    $path $_model --batch-size $bs --accumulation_steps $accum \
    --fishpp 1 --fish_global_heads 3 --fish_mask_levels 3 --fish_mask_type dist \
    --fish_non_linear 0 --fish_non_linear_bias 0 \
    --fish_cls_token_type pos --fish_cls_token_pos 0.5 \
    --fish_global_proj_type mix --fish_layer_limit 8

