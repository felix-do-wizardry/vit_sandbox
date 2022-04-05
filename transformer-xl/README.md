# FISH++ with LM - WikiText-3

## Installation

### Data
Get to the dataset root path then run:
```bash
wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip -q wikitext-103-v1.zip
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..
```
### Dependencies

```bash
conda install pytorch torchvision -c pytorch
```

### Train
```bash
cd pytorch

# bash run_wt103_base_custom.sh train --work_dir /host/ubuntu/lm
```

> wt103
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --dataset wt103 \
    --data /host/ubuntu/data/lm/wikitext-103 --work_dir /host/ubuntu/lm/fishpp \
    --adaptive --n_layer 16 --d_model 410 --n_head 10 --d_head 41 --d_inner 2100 \
    --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 0 --attn_type 2 \
    --max_step 200000 --eval-interval 2000 --mem_len 0 --tgt_len 160 --eval_tgt_len 160 \
    --batch_size 60 --wandb 1 --exp_name baseline \


CUDA_VISIBLE_DEVICES=0 python train.py --cuda --dataset wt103 \
    --data /host/ubuntu/data/lm/wikitext-103 --work_dir /host/ubuntu/lm/fishpp \
    --adaptive --n_layer 16 --d_model 410 --n_head 10 --d_head 41 --d_inner 2100 \
    --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 0 --attn_type 2 \
    --max_step 200000 --eval-interval 2000 --mem_len 0 --tgt_len 160 --eval_tgt_len 160 \
    --batch_size 60 --wandb 1 --exp_name fishpp_h3_g2m \
    --fishpp 1 --mask_levels 3 --global_heads 2 --global_proj_type mix \
    --layer_limit -1 --layer_offset 0 --non_linear 0 --non_linear_bias 1


# TEST
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --dataset wt103 \
    --data /host/ubuntu/data/lm/wikitext-103 --work_dir /host/ubuntu/lm/fishpp \
    --adaptive --n_layer 16 --d_model 410 --n_head 10 --d_head 41 --d_inner 2100 \
    --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 0 --attn_type 2 \
    --max_step 200000 --eval-interval 200 --mem_len 0 --tgt_len 160 --eval_tgt_len 160 \
    --batch_size 60 --wandb 1 --exp_name fishpp_h3_g2m \
    --fishpp 1 --mask_levels 3 --global_heads 2 --global_proj_type mix \
    --layer_limit -1 --layer_offset 0 --non_linear 0 --non_linear_bias 1


```

> wt103_B ??
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --dataset wt103 \
    --data /host/ubuntu/data/lm/wikitext-103 --work_dir /host/ubuntu/lm/fishpp \
    --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 \
    
    --multi_gpu --use_wandb --project_name 'mgk'
```
