""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from backbones.h_matrix import H_Matrix, H_Matrix_Masks, MASK_TYPES
import numpy as np

_logger = logging.getLogger(__name__)

# %%
DEBUG = 1
# FISH_MASK_TYPES = [
#     # 'h1d',
#     'h',
#     'hdist',
#     'dist',
#     'distq',
# ]

# %%
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


# %%
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if DEBUG:
            print(f'<Attention> [BASE] heads[{num_heads}] qkv[{dim}->{dim*3}]')
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_FishPP(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                global_heads=1,
                mask_type='h',
                mask_levels=3,
                token_grid_size=1,
                masks=None,
                mask_base=None,
                non_linear=0,
                non_linear_bias=1,
                
                # global_full_proj=1,
                # global_full_mix=0,
                # global_mix_type='full',
                
                global_proj_type='full',
                
                metrics_test=False,
                # **kwargs,
                ):
        super().__init__()
        
        self.global_heads = global_heads
        self.mask_type = mask_type
        self.mask_levels = mask_levels
        self.token_grid_size = token_grid_size
        self.token_count = self.token_grid_size * self.token_grid_size + 1
        # self.global_full_proj = bool(global_full_proj)
        # self.global_full_mix = bool(global_full_mix)
        assert global_proj_type in ['base', 'full', 'mix']
        self.global_proj_type = global_proj_type
        
        self.num_heads = num_heads
        assert self.num_heads % self.global_heads == 0
        
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        global_dim = int(head_dim) * self.global_heads
        self.global_dim = global_dim
        self.total_heads = self.global_heads * 2 + self.num_heads
        total_dim = self.head_dim * self.total_heads
        
        self.qkv = nn.Linear(dim, total_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # self.pi = nn.Parameter(torch.ones(1, 1, self.mask_levels))
        head_ratio = int(self.num_heads // self.global_heads)
        self.head_ratio = head_ratio
        # self.level_mix = nn.Linear(self.mask_levels, self.num_heads // self.global_heads, bias=False)
        # torch.nn.init.constant_(self.level_mix.weight, 1.0)
        
        # global mix type: mask_weights
        # - base: [1, HR]
        # - full: [G, HR]
        # - mix : [G, HR]
        if self.global_proj_type == 'mix':
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.num_heads * self.global_heads, device='cuda'))
        elif self.global_proj_type == 'full':
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, self.num_heads, device='cuda'))
        else:
            self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, head_ratio, device='cuda'))
        
        self.masks = masks
        self.mask_base = mask_base
        
        self.is_non_linear = non_linear
        if non_linear:
            self.head_proj = nn.Linear(self.num_heads, self.num_heads, bias=non_linear_bias)
        else:
            self.head_proj = None
        
        self.metrics_test = bool(metrics_test)
        if self.metrics_test:
            # FishPP:
            # masks [1, 1, N, N, Level] -> [1, 1, N, N, HR / H / GH*H]
            # mask_base [1, 1, N, N, 1]
            mask_weights = self.masks @ self.mask_proj
            mask_weights = mask_weights + self.mask_base
            
            N = self.token_count
            if self.global_proj_type == 'mix':
                mask_weights = torch.ones(1, self.global_heads, N, N, self.num_heads, device='cuda')
                # masks_weights [1, GH, N, N, H]
            elif self.global_proj_type == 'full':
                mask_weights = torch.ones(1, self.global_heads, N, N, self.head_ratio, device='cuda')
                # masks_weights [1, GH, N, N, HR]
            else:
                mask_weights = torch.ones(1, 1, N, N, self.head_ratio, device='cuda')
                # masks_weights [1, 1, N, N, HR]
            self._mask_weights = mask_weights
        
        if DEBUG:
            print(f'<Attention> [FISH]',
                f'heads[{global_heads}->{num_heads}]',
                f'mask_type[{mask_type}]',
                f'qkv[{dim}->{total_dim}]',
                f'levels[{mask_levels}]',
                f't[{token_grid_size}]',
                # f'global[{"mix" if global_full_mix else ("full" if global_full_proj else "base")}]',
                f'global[{self.global_proj_type}]',
                f'masks_shape{list(masks.shape)}',
                ' [METRICS_TEST]' if self.metrics_test else '',
            )
    
    def forward_test(self, x):
        B, N, C = x.shape
        
        # -> [B, N, 2*global_dim + dim]
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.total_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        q = qkv[:, : self.global_heads]
        k = qkv[:, self.global_heads : self.global_heads * 2]
        v = qkv[:, self.global_heads * 2 : ]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # FishPP:
        # [1, 1, N, N, HR / H / GH*H]
        mask_weights = self._mask_weights
        
        # attn [B, GH, N, N] x masks [1, 1/GH, N, N, HR/H] -> [B, GH, N, N, HR/H]
        attn = attn[..., None] * mask_weights
        
        if self.global_proj_type == 'mix':
            # [B, GH, N, N, H] -> [B, N, N, H]
            attn = torch.sum(attn, dim=-4, keepdim=False)
            # [B, N, N, H] -> [B, N, N, GH, HR]
            attn = attn.reshape([B, N, N, self.global_heads, self.head_ratio])
            # [B, N, N, GH, HR] -> [B, GH, N, N, HR]
            attn = attn.permute(0, 3, 1, 2, 4)
        
        # attn: [B, GH, N, N, HR]
        
        if self.is_non_linear:
            # changed from HR->HR to H->H
            # attn: [B, GH, N, N, HR] -> [B, N, N, H]
            attn = attn.permute(0, 2, 3, 1, 4).contiguous().reshape(B, N, N, self.num_heads)
            attn = nn.functional.relu(attn)
            attn = self.head_proj(attn)
            # [B, N, N, H] -> [B, H, N, N]
            attn = attn.permute(0, 3, 1, 2).contiguous()
        else:
            # [B, GH, N, N, HR] -> [B, GH, HR, N, N] -> [B, H, N, N]
            attn = attn.permute(0, 1, 4, 2, 3).contiguous().reshape(B, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward(self, x):
        if self.metrics_test:
            return self.forward_test(x)
        
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # -> [B, N, 2*global_dim + dim]
        qkv = self.qkv(x)
        
        # qkv = qkv.reshape(B, N, 3, self.global_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        qkv = qkv.reshape(B, N, self.total_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        q = qkv[:, : self.global_heads]
        k = qkv[:, self.global_heads : self.global_heads * 2]
        v = qkv[:, self.global_heads * 2 : ]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # FishPP:
        # masks [1, 1, N, N, Level] -> [1, 1, N, N, HR / H / GH*H]
        # mask_base [1, 1, N, N, 1]
        mask_weights = self.masks @ self.mask_proj
        mask_weights = mask_weights + self.mask_base
        
        if self.global_proj_type == 'mix':
            mask_weights = mask_weights.reshape([1, N, N, self.global_heads, self.num_heads])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, H]
        elif self.global_proj_type == 'full':
            mask_weights = mask_weights.reshape([1, N, N, self.global_heads, self.head_ratio])
            mask_weights = mask_weights.permute(0, 3, 1, 2, 4)
            # masks_weights [1, GH, N, N, HR]
        
        # attn [B, GH, N, N] x masks [1, 1/GH, N, N, HR/H] -> [B, GH, N, N, HR/H]
        attn = attn[..., None] * mask_weights
        
        if self.global_proj_type == 'mix':
            # [B, GH, N, N, H] -> [B, N, N, H]
            attn = torch.sum(attn, dim=-4, keepdim=False)
            # [B, N, N, H] -> [B, N, N, GH, HR]
            attn = attn.reshape([B, N, N, self.global_heads, self.head_ratio])
            # [B, N, N, GH, HR] -> [B, GH, N, N, HR]
            attn = attn.permute(0, 3, 1, 2, 4)
        
        # attn: [B, GH, N, N, HR]
        
        if self.is_non_linear:
            # changed from HR->HR to H->H
            # attn: [B, GH, N, N, HR] -> [B, N, N, H]
            attn = attn.permute(0, 2, 3, 1, 4).contiguous().reshape(B, N, N, self.num_heads)
            attn = nn.functional.relu(attn)
            attn = self.head_proj(attn)
            # [B, N, N, H] -> [B, H, N, N]
            attn = attn.permute(0, 3, 1, 2).contiguous()
        else:
            # [B, GH, N, N, HR] -> [B, GH, HR, N, N] -> [B, H, N, N]
            attn = attn.permute(0, 1, 4, 2, 3).contiguous().reshape(B, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# %%
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fishpp=False,
                 **kwargs,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if fishpp:
            self.attn = Attention_FishPP(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                **kwargs,
            )
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# TODO: add argument for cls_mask settings
# %%
class VisionTransformer_FishPP(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 global_heads=1,
                 mask_type='dist',
                 mask_levels=3,
                 non_linear=0,
                 non_linear_bias=1,
                 
                #  global_full_proj=0,
                #  global_full_mix=0,
                 global_proj_type='full',
                 cls_token_pos=1,
                 cls_token_type='mask',
                 
                 layer_limit=None,
                 layer_offset=0,
                 
                 metrics_test=False,
                 **kwargs
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if len(kwargs) > 0:
            print('overflown kwargs:', kwargs)
        
        assert mask_levels >= 2
        self.mask_levels = mask_levels
        self.h_levels = self.mask_levels - 1
        
        assert isinstance(global_heads, int) and global_heads >= 1
        self.global_heads = global_heads
        assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] not in {MASK_TYPES}'
        self.mask_type = mask_type
        
        self.token_grid_size = int(img_size // patch_size)
        self.token_count = self.token_grid_size * self.token_grid_size + 1
        
        # assert cls_token_type in ['copy', 'pos', 'mask', 'sum'], f'cls_token_type[{cls_token_type}] is invalid'
        self.cls_token_type = cls_token_type
        
        assert global_proj_type in ['base', 'full', 'mix'], f'global_proj_type[{global_proj_type}] is invalid'
        self.global_proj_type = global_proj_type
        
        # if cls_token_pos >= 0 and self.mask_type == 'dist':
        if self.cls_token_type == 'pos':
            assert self.mask_type in ['dist', 'distq']
            assert isinstance(cls_token_pos, (float, int))
            assert 0 <= cls_token_pos <= 1
            self.cls_token_pos = float(np.clip(cls_token_pos, 0, 1))
        else:
            self.cls_token_pos = None
        
        assert layer_offset >= 0
        layer_range = [layer_offset, depth]
        # if layer_offset >= 0:
        #     layer_range[0] = layer_offset
        #     # fishpp_layers = list(range(layer_offset, depth))
        if layer_limit is not None and layer_limit >= 1:
            layer_range[1] = layer_offset + layer_limit
            # fishpp_layers = list(range(layer_offset, layer_offset + layer_limit))
        
        # h_levels == mask_levels - 1
        # hm = H_Matrix(
        #     t=self.token_grid_size,
        #     level=self.h_levels,
        #     mask_type=self.mask_type,
        #     cls_token_count=1,
        #     cls_token_order='first',
        #     cls_token_pos=self.cls_token_pos,
        # )
        
        # indexed_mask = hm.indexed_mask
        # mask_levels_final = self.mask_levels
        
        # # [N, N] -> [N, N] -> [1, 1, N, N, 1]
        # mask_base = torch.tensor(
        #     indexed_mask * 0,
        #     dtype=torch.float32,
        #     device='cuda',
        #     requires_grad=False,
        # )[None, None, :, :, None]
        
        # _shape = list(indexed_mask.shape)
        # assert (len(_shape) == 2 and _shape[0] == _shape[1] == self.token_count
        #     ), f'indexed_mask.shape={_shape} != token_count[{self.token_count}]'
        
        # if self.cls_token_type in ['copy']:
        #     # [copy] not using cls token projection -> set mask_base to 1 for -1 values (cls)
            
        #     # [N, N] -> [N, N] -> [1, 1, N, N, 1]
        #     mask_base = torch.tensor(
        #         indexed_mask == -1,
        #         dtype=torch.float32,
        #         device='cuda',
        #         requires_grad=False,
        #     )[None, None, :, :, None]
            
        #     # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
        #     masks = torch.tensor(np.array([
        #         indexed_mask == i
        #         for i in range(self.mask_levels)
        #     ]), dtype=torch.float32,
        #         device='cuda',
        #         requires_grad=False,
        #     ).permute(1, 2, 0)[None, None]
            
        # elif self.cls_token_type in ['pos']:
        #     assert np.all(indexed_mask >= 0), 'dist pos ?? cls mask (should be >= 0)'
            
        #     # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
        #     masks = torch.tensor(np.array([
        #         indexed_mask == i
        #         for i in range(self.mask_levels)
        #     ]), dtype=torch.float32,
        #         device='cuda',
        #         requires_grad=False,
        #     ).permute(1, 2, 0)[None, None]
        
        # elif self.cls_token_type in ['mask']:
        #     # [mask] using cls token projection -> set masks as new value for -1 in indexed_mask
        #     assert np.all(indexed_mask[0, :] == -1), '?? cls mask (should be -1)'
        #     assert np.all(indexed_mask[:, 0] == -1), '?? cls mask (should be -1)'
        #     assert np.all(indexed_mask[1:, 1:] >= 0), '?? non-cls mask (should be >= 0)'
            
        #     # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
        #     masks = torch.tensor(np.array([
        #         indexed_mask == i
        #         for i in [*list(range(self.mask_levels)), -1]
        #     ]), dtype=torch.float32,
        #         device='cuda',
        #         requires_grad=False,
        #     ).permute(1, 2, 0)[None, None]
        #     mask_levels_final = self.mask_levels + 1
            
        # elif self.cls_token_type in ['sum']:
        #     # [sum] using cls token projection -> set masks to all True for -1 in indexed_mask
        #     assert np.all(indexed_mask[0, :] == -1), '?? cls mask (should be -1)'
        #     assert np.all(indexed_mask[:, 0] == -1), '?? cls mask (should be -1)'
        #     assert np.all(indexed_mask[1:, 1:] >= 0), '?? non-cls mask (should be >= 0)'
            
        #     # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
        #     masks = torch.tensor(np.array([
        #         np.any([indexed_mask == i, indexed_mask == -1], axis=0)
        #         for i in range(self.mask_levels)
        #     ]), dtype=torch.float32,
        #         device='cuda',
        #         requires_grad=False,
        #     ).permute(1, 2, 0)[None, None]
        
        masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks(
            mask_type=self.mask_type,
            cls_token_type=self.cls_token_type,
            cls_token_pos=self.cls_token_pos,
            cls_token_count=1,
            token_grid_size=self.token_grid_size,
            mask_levels=self.mask_levels,
        )
        masks = torch.tensor(
            masks,
            dtype=torch.float32, device='cuda', requires_grad=False,)
        mask_base = torch.tensor(
            mask_base,
            dtype=torch.float32, device='cuda', requires_grad=False,)
        
        print()
        print(f'<VIT> [FISHPP]',
            f'depth[{depth}]',
            f't[{self.token_grid_size}]',
            f'masks[{masks.size()}]',
            f'mask_base[{mask_base.size()}]',
            f'mask_type[{self.mask_type}]',
            # f'global[{"mix" if global_full_mix else ("full" if global_full_proj else "base")}]',
            f'global[{self.global_proj_type}]',
            f'layer_range[{layer_range[0]}-{layer_range[1]}]',
        )
        print()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # TODO: implement cls_token_type & global_proj_type
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                fishpp=layer_range[0] <= i < layer_range[1],
                global_heads=self.global_heads,
                mask_type=self.mask_type,
                token_grid_size=int(img_size // patch_size),
                masks=masks,
                mask_base=mask_base,
                mask_levels=mask_levels_final,
                non_linear=non_linear,
                non_linear_bias=non_linear_bias,
                # global_full_proj=global_full_proj,
                # global_full_mix=global_full_mix,
                
                global_proj_type=self.global_proj_type,
                
                metrics_test=metrics_test,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        print()

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# %%
class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 **kwargs
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        if len(kwargs) > 0:
            print('DeiT overflown kwargs:', kwargs)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# %%
def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def deit_small_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def deit_base_patch16_224(pretrained=False, **kwargs):
#     """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def deit_base_patch16_384(pretrained=False, **kwargs):
#     """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer(
#         'deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
#     return model


# @register_model
# def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer(
#         'deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
#     return model


# @register_model
# def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
#     """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(
#         'deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
#     return model


# @register_model
# def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
#     """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
#     ImageNet-1k weights from https://github.com/facebookresearch/deit.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer(
#         'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
#     return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model