# CUDA_VISIBLE_DEVICES='0,2,5,7' python -m torch.distributed.launch --master_port 311 --nproc_per_node=4 \
# tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py --resume-from /tam/data/coco/swin_chks/swim_gmm3x_new/latest.pth --work-dir /tam/data/coco/swin_chks/swim_gmm3x_new_resume --launcher pytorch ${@:3}
# CUDA_VISIBLE_DEVICES=4 python tools/test.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py /tam/data/coco/swin_chks/swim_gmm3x/latest.pth --eval bbox segm 
# CUDA_VISIBLE_DEVICES='0,1,2,5,3,4,6,7' python -m torch.distributed.launch --master_port 312 --nproc_per_node=8 \
# tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir /tam/data/coco/swin_chks/swim_gmm1x_prune_resume --resume-from /tam/data/coco/swin_chks/swim_gmm1x_prune/latest.pth --launcher pytorch ${@:3}

# CUDA_VISIBLE_DEVICES='0,1,6,7' python -m torch.distributed.launch --master_port 311 --nproc_per_node=4 \
# tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py --work-dir /tam/data/coco/swin_chks/swim_gmm3x_prune --launcher pytorch ${@:3}
CUDA_VISIBLE_DEVICES='0,1,2,5,3,4,6,7' python -m torch.distributed.launch --master_port 312 --nproc_per_node=8 \
tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --work-dir /tam/data/coco/swin_chks/swim_gmm1x_prune_global_local07 --launcher pytorch ${@:3}


# # metrics
# CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --master_port 400 --nproc_per_node=1 \
#     tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py \
#     --work-dir /tam/data/coco/swin_chks/swim_gmm1x_prune_global_local07 --launcher pytorch ${@:3}