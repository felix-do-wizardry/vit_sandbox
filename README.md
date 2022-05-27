# fiak_sandbox

## FiAK Metrics
> deit_tiny with 4 heads

> CONFIGS
```bash
data_path="/docker/code/data/imagenet2012"
output_path="/docker/code/data/vision/fishsv"
path="--data-path "$data_path" --output_dir "$output_path
gpus="7"
gpu_count=1
_cuda="-m torch.distributed.launch --nproc_per_node="$gpu_count" --use_env --master_port"
bs_test=256


_model="--model deit_tiny_patch16_224 --metrics_only 1 --wandb 1"
bs=256
batch_limit=40


_model="--model deit_tiny_patch8_224 --metrics_only 1 --wandb 1"
bs=64
batch_limit=40


_model="--model deit_tiny_patch4_224 --metrics_only 1 --wandb 1"
bs=4
batch_limit=40

```

> TRAIN: base + fiak + gmm
```bash
CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12000 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode train --attn_type base

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12001 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode train --attn_type fiak

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12002 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode train --attn_type gmm



# > METRIC: base + fiak_0.7_0.2 + fiak_0.7_0.7 + gmm_0.3_0.3
CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12101 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric --attn_type base

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12101 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric --attn_type fiak --prune_total 0.7 --prune_k 0.2

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12102 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric --attn_type fiak --prune_total 0.7 --prune_k 0.7

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12102 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric --attn_type gmm --prune_total 0.3 --prune_k 0.3



# > METRIC_K: base + fiak_0.7_0.2 + fiak_0.7_0.7 + gmm_0.3_0.3
CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12201 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_k --attn_type fiak --prune_total 0.7 --prune_k 0.2

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12202 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_k --attn_type fiak --prune_total 0.7 --prune_k 0.7

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12202 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_k --attn_type gmm --prune_total 0.3 --prune_k 0.3


# > METRIC_MM: base + fiak_0.7_0.2 + fiak_0.7_0.7 + gmm_0.3_0.3
# (matmul instead of dist for fiak and gmm)
CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12301 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_mm --attn_type fiak --prune_total 0.7 --prune_k 0.2

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12302 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_mm --attn_type fiak --prune_total 0.7 --prune_k 0.7

CUDA_VISIBLE_DEVICES=$gpus python $_cuda 12302 main.py \
    $path $_model --batch_limit $batch_limit --batch-size $bs \
    --mode metric_mm --attn_type gmm --prune_total 0.3 --prune_k 0.3

```















> TRAIN: base + fiak_0.5_0.2
```bash
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12000 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode train \
    --attn_type base

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12001 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode train \
    --attn_type fiak --prune_total 0.5 --prune_k 0.2

```

> METRIC: base + fiak_0.5_0.2
```bash
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 13000 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode metric \
    --attn_type base

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 13001 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode metric \
    --attn_type fiak --prune_total 0.5 --prune_k 0.2

```

> METRIC_K: base + fiak_0.5_0.2
```bash
CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 13000 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode metric_k \
    --attn_type base

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 13001 main.py \
    --data-path /docker/code/data/imagenet2012 --output_dir /docker/code/data/vision/fiak \
    --model deit_tiny_patch16_224 --metrics_only 1 --wandb 1 \
    --batch_limit 40 --batch-size 256 --mode metric_k \
    --attn_type fiak --prune_total 0.5 --prune_k 0.2

```
