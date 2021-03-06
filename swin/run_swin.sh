# installation
cd swin
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..


# Baseline
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12800 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_baseline.yaml


# A0 dist3 gr3m
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12801 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_dist3_gr3m.yaml


# A1 distq3 gr3m
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12802 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_distq3_gr3m.yaml


# A2 cross3 gr3m
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12803 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_cross3_gr3m.yaml

# A3 crossq3 gr3m
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12804 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_crossq3_gr3m.yaml


# D0 dist3 gr3m s3
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12900 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_dist3_gr3m_s3.yaml


# D1 cross3 gr3m s3
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12901 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin \
    --batch-size 128 --accumulation-steps 8 \
    --cfg configs/swinT_cross3_gr3m_s3.yaml

















# quick test
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12799 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin_temp \
    --batch-size 8 --accumulation-steps 8 \
    --cfg configs/swinT_dist3_gr3m.yaml --wandb 0

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12799 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin_temp \
    --batch-size 8 --accumulation-steps 8 \
    --cfg configs/swinT_distq3_gr3m.yaml --wandb 0

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12799 main.py \
    --data-path /host/ubuntu/data/imagenet2012 --output /host/ubuntu/vision/fishpp_swin_temp \
    --batch-size 8 --accumulation-steps 8 \
    --cfg configs/swinT_cross3_gr3m.yaml --wandb 0





