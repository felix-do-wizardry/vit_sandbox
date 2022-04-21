# Cross-Covariance Image Transformer (XCiT)

> fishpp single gpu
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
  --data-path /docker/code/data/imagenet2012 --output_dir ./experiments/fishpp/xcit_tiny_12_p16 \
  --epochs 400 --model xcit_tiny_12_p16 --drop-path 0.05 \
  --batch-size 256 --accumulation_steps 4 \


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
  --data-path /docker/code/data/imagenet2012 --output_dir ./experiments/fishpp/xcit_tiny_12_p16_h3_g1m \
  --epochs 400 --model xcit_tiny_12_p16 --drop-path 0.05 \
  --batch-size 32 --accumulation_steps 2 \
  --fishpp 1 --fish_non_linear 0 --fish_non_linear_bias 1 \
  --fish_global_heads 1 --fish_mask_level 3 --fish_mask_type h \
  --fish_global_proj_type mix --fish_layer_limit -1 --wandb 1


```


> fishpp quick runs single gpu
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
  --data-path /host/ubuntu/data/imagenet2012 --output_dir ./experiments/fishpp/xcit_tiny_12_p16 \
  --epochs 400 --model xcit_tiny_12_p16 --drop-path 0.05 \
  --batch-size 32 --accumulation_steps 2 \

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
  --data-path /host/ubuntu/data/imagenet2012 --output_dir ./experiments/fishpp/xcit_tiny_12_p16_h3_g1m \
  --epochs 400 --model xcit_tiny_12_p16 --drop-path 0.05 \
  --batch-size 32 --accumulation_steps 2 \
  --fishpp 1 --fish_non_linear 0 --fish_non_linear_bias 1 \
  --fish_global_heads 1 --fish_mask_level 3 --fish_mask_type h \
  --fish_global_proj_type mix --fish_layer_limit -1 --wandb 1
  
```
