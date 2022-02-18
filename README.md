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
    --output_dir ./outputs/ \
```

> deitB_p16_224_fishpp_hl2_bs256x4
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 256 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --output_dir /path/to/save \
```



# QUICK TEST

> deitB_p16_224_baseline_bs128x1
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 128 \
    --data-path /host/ubuntu/data/imagenet2012 \
    --output_dir ./outputs/ \
```

> deitB_p16_224_fishpp
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 --batch-size 2 \
    --data-path /host/ubuntu/data/imagenet2012 --output_dir ./outputs/ \
    --fishpp 1 --fish_global_heads 1 --fish_mask_type hdist --fish_mask_levels 3 \

```
