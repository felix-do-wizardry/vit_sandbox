# vit_sandbox

## FISH++
Features/ideas:
- uses 1 or few global heads
- mixes attention matrix based on position pairs of QK
- 
### FISH++ DeiT
Notes:
- working dir: [deit](./deit)
- added on 220216

Baseline:
- deit_base_patch16_224 - acc 81.8% / 95.6%
- head = 12
- batch_size = 256 x 4 = 1024

Fish++:
- global_head = 1
- H_level = 2
- mask_count = H_level + 1 = 3
- pi projection: 3 -> 12
- non-linear = True/False

> deitB_p16_224_baseline_bs256x4
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --output_dir /host/ubuntu/vision/fishpp \
```

> deitB_p16_224_fishpp_hl2_bs256x4
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --model deit_base_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type hdist --fish_mask_levels 3 \
```



# QUICK TEST

> deitB_p16_224_baseline_bs128x1
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 128 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --output_dir /host/ubuntu/vision/fishpp \
```

> deitB_p16_224_fishpp_bs128x1
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 --batch-size 128 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type hdist --fish_mask_levels 3 \

```

> deitS_p16_224_baseline_bs256x1
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp
```

> deitS_p16_224_fishpp_bs256x1
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type hdist --fish_mask_levels 3 \

```

> deitS_p16_224_fishpp_bs256x1x4 | accumulating 4 steps
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type hdist --fish_mask_levels 3 \
    --accumulation_steps 4

```

> deitS_p16_224_fishpp_bs256x1x4 | dist | accumulating 4 steps
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type dist --fish_mask_levels 3 \
    --accumulation_steps 4


python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_small_patch16_224 --batch-size 128 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp \
    --fishpp 1 --fish_global_heads 2 --fish_mask_type hdist --fish_mask_levels 3 \
    --accumulation_steps 8


```

# ACCUMULATION
```
echo METRICS tiny baseline 256
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12347 main.py \
    --model deit_tiny_patch16_224 --batch-size 64 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_temp \
    --batch_limit 100 --accumulation_steps 4

python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port 12347 main.py \
    --model deit_tiny_patch16_224 --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir /host/ubuntu/vision/fishpp_temp \
    --batch_limit 100 --accumulation_steps 0

```