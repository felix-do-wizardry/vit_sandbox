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

bash run_wt103_base_custom.sh train --work_dir /host/ubuntu/lm
```

```bash
python train.py --cuda --dataset wt103 \
    --data /host/ubuntu/data/lm/wikitext-103 --work_dir /host/ubuntu/lm/fishpp \
    --adaptive --n_layer 16 --d_model 410 --n_head 10 --d_head 41 --d_inner 2100 \
    --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 0 \
    --max_step 200000 --mem_len 160 --tgt_len 160 --eval_tgt_len 160 \
    --batch_size 60 --wandb 1 --exp_name baseline \



    # --multi_gpu \
    # --gpu0_bsz 4 \

```

