# TODO: implement universal fishpp attention for all 2D vision transformer architectures

# %%
import torch
import torch.nn as nn
# import pandas as pd
import numpy as np
import time, os, re, string, json
from timm.utils import NativeScaler as timm_NativeScaler
from timm.utils import dispatch_clip_grad

# %%
class Global_QKV(nn.Module):
    def __init__(self, dim, local_heads=8, global_heads=None, qkv_bias=False):
        super().__init__()
        self.local_heads = local_heads
        self.global_heads = global_heads
        head_dim = dim // local_heads
        self.head_dim = head_dim
        
        # self.scale = qk_scale or head_dim ** -0.5
        
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if global_heads is None:
            global_heads = local_heads
        dim_global = int(dim / local_heads * global_heads)
        
        self.proj_q = nn.Linear(dim, dim_global, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim_global, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        print('<Global_QKV> [FISHPP]')
    
    def forward(self, x):
        # REFERENCE:
        # B, N, C = x.shape
        # # [B, N, C] -> [B, N, 3C] -> [B, N, 3, H, D]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # # [B, N, 3, H, D] -> [3, B, H, N, D]
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # # [B, H, N, D]
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # return q, k, v
        
        # CUSTOM:
        B, N, C = x.shape
        q = self.proj_q(x).reshape(B, N, self.global_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).reshape(B, N, self.global_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(x).reshape(B, N, self.local_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # q/k [B, GH, N, D]
        # v [B, H, N, D]
        return q, k, v

# %%
class Global_Attn(nn.Module):
    global_proj_types = ['mix', 'full']
    
    def __init__(self,
                masks,
                mask_levels,
                local_heads,
                global_heads=None,
                global_proj_type='mix',
                non_linear=False,
                non_linear_bias=True,
                device='cuda',
                # **kwargs,
                ):
        super().__init__()
        self.local_heads = local_heads
        if global_heads is None:
            self.global_heads = local_heads
        else:
            self.global_heads = global_heads
        self.head_ratio = int(self.local_heads // self.global_heads)
        
        self.masks = masks
        self.mask_levels = mask_levels
        
        assert global_proj_type in self.global_proj_types, f'global_proj_type[{global_proj_type}] must be one of {self.global_proj_types}'
        self.global_proj_type = global_proj_type
        if global_proj_type == 'mix':
            mask_proj_units = [self.mask_levels, self.local_heads * self.global_heads]
        elif global_proj_type == 'full':
            mask_proj_units = [self.mask_levels, self.local_heads]
        else:
            # [DEPRECATED] assume base
            mask_proj_units = [self.mask_levels, self.head_ratio]
        self.mask_proj = nn.Parameter(torch.ones(*mask_proj_units, device=device))
        
        self.is_non_linear = non_linear
        if non_linear:
            self.head_proj = nn.Linear(self.local_heads, self.local_heads, bias=non_linear_bias)
        else:
            self.head_proj = None
        print('<Global_Attn> [FISHPP]')
        
        
    def forward(self, q, k, attn_bias_add=None, temperature=None):
        '''Get attn from q and k, then mix the global attention matrices into local attentions
        (return pre-temperaturized and pre-softmax attn values)
        
        Input: q/k (tensor) shape=[B, GH, N, D]
        Output: attn (tensor) shape=[B, H, N, N]
        '''
        
        B, GH, Nq, D = q.shape
        Bk, GHk, Nk, Dk = k.shape
        # assert B == Bk
        # assert GH == GHk
        # assert D == Dk
        
        # q = q * self.scale
        
        # [B, H, N, N] (applicable to XCA as well [B, H, D, D])
        attn = (q @ k.transpose(-2, -1))
        
        if temperature is not None:
            attn = attn * temperature
        
        # (for swin) attn_bias_add == relative_position_bias.unsqueeze(0)
        if attn_bias_add is not None:
            attn = attn + attn_bias_add

        # FishPP:
        # masks [1, 1, N, N, Level] -> mask_weights [1, N, N, 1, H / GH*H]
        # version D: changed target mask_weights.shape from [1, 1/GH, N, N, HR/H] to [1, N, N, GH, HR/H]
        mask_weights = self.masks @ self.mask_proj
        
        # [DEPRECATED] version B: attn [B, GH, N, N] x masks [1, 1/GH, N, N, HR/H] -> [B, GH, N, N, HR/H]
        # version D: changed target attn.shape from [B, GH, N, N, HR] to [B, N, N, H]
        
        if self.global_proj_type == 'mix':
            mask_weights = mask_weights.reshape([1, Nq, Nk, self.global_heads, self.local_heads])
            # attn [B, N, N, GH, 1] x mask_weights [1, N, N, GH, H] -> [B, N, N, GH, H]
            attn = attn.permute(0, 2, 3, 1).unsqueeze(-1) * mask_weights
            # [B, N, N, GH, H] -> [B, N, N, H]
            attn = torch.sum(attn, dim=-2, keepdim=False)
            
        elif self.global_proj_type == 'full':
            mask_weights = mask_weights.reshape([1, Nq, Nk, self.global_heads, self.head_ratio])
            # attn [B, N, N, GH, 1] x mask_weights [1, N, N, GH, HR] -> [B, N, N, GH, HR]
            attn = attn.permute(0, 2, 3, 1).unsqueeze(-1) * mask_weights
            # [B, N, N, GH, HR] -> [B, N, N, H]
            attn = attn.reshape([B, Nq, Nk, self.local_heads])
        
        # ver[D] attn: [B, N, N, H]
        
        if self.is_non_linear:
            # version B: projection changed from HR->HR to H->H
            # attn [B, N, N, H]
            attn = attn.contiguous()
            attn = nn.functional.relu(attn)
            attn = self.head_proj(attn)
        
        # [B, N, N, H] -> [B, H, N, N]
        attn = attn.permute(0, 3, 1, 2).contiguous()
        
        # [B, H, N, N]
        return attn

# %%
class Experiment:
    def __init__(self,
                model='unknown',
                fishpp=True,
                mask_type='dist',
                mask_levels=3,
                global_heads_str='3',
                global_proj_str='m',
                bs=128,
                gpu=1,
                accumulation_steps=1,
                time_stamp=None,
                ):
        if time_stamp is None:
            self.time_stamp = time.strftime('%y%m%d_%H%M%S')
        else:
            self.time_stamp = str(time_stamp)
        
        assert isinstance(bs, int) and bs >= 1
        assert isinstance(gpu, int) and gpu >= 1
        assert isinstance(accumulation_steps, int) and accumulation_steps >= 0
        if accumulation_steps < 2:
            accumulation_steps = 1
        self.accumulation_steps = accumulation_steps
        self.gpu = gpu
        self.bs = bs
        self.bs_eff = bs * gpu * accumulation_steps
        
        self.mask_type = str(mask_type)
        self.mask_levels = str(mask_levels)
        self.global_heads_str = str(global_heads_str)
        self.global_proj_str = str(global_proj_str)
        
        self.fishpp = bool(fishpp)
        self.name_bs = f'{self.bs}x{self.gpu}{"" if self.accumulation_steps < 2 else "x"+str(self.accumulation_steps)}'
        self.model = str(model)
        if self.fishpp:
            self.name_type = f'{self.mask_type}{self.mask_levels}_g{self.global_heads_str}{self.global_proj_str}'
        else:
            self.name_type = 'baseline'
        
        self.name = f'{self.model}_{self.name_type}_{self.name_bs}'
        self.exp_name = f'{self.time_stamp}_{self.name}'
        
    
    def __repr__(self) -> str:
        return f'ts[{self.time_stamp}] type[{self.name_type}] name[{self.name}] exp_name[{self.exp_name}]'

# Experiment(
#     # fishpp=False,
#     fishpp=True,
#     model='swin',
#     mask_type='dist',
#     mask_levels=3,
#     global_heads_str='r3',
#     global_proj_str='m',
#     bs=256,
#     gpu=2,
#     accumulation_steps=2,
# )

# %%
class NativeScaler(timm_NativeScaler):
    state_dict_key = "amp_scaler"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self,
                loss,
                optimizer,
                clip_grad=None,
                clip_mode='norm',
                parameters=None,
                create_graph=False,
                with_step_update=True,
                ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if not with_step_update:
            return
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()
        
    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


