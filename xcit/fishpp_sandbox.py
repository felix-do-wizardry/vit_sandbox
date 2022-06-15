# TODO: implement universal fishpp attention for all 2D vision transformer architectures

# %%
import torch
import torch.nn as nn
# import pandas as pd
import numpy as np
import time, os, re, string, json

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
class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # CLS attention does not support fishpp right now! (global_heads=None)
        self.global_qkv = Global_QKV(
            dim=dim,
            local_heads=num_heads,
            global_heads=None,
            qkv_bias=qkv_bias,
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        q, k, v = self.global_qkv(x)
        
        
        qc = q[:, :, 0:1]   # CLS token
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        
        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_tkn = self.proj(cls_tkn)
        x = torch.cat([self.proj_drop(cls_tkn), x[:, 1:]], dim=1)
        return x


# %% DeiT-based Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                masks=None,
                mask_levels=3,
                global_heads=3,
                global_proj_type='mix',
                non_linear=False,
                ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.global_qkv = Global_QKV(
            dim=dim,
            local_heads=num_heads,
            global_heads=global_heads,
            qkv_bias=qkv_bias,
        )
        self.global_attn = Global_Attn(
            masks=masks,
            mask_levels=mask_levels,
            local_heads=num_heads,
            global_heads=global_heads,
            global_proj_type=global_proj_type,
            non_linear=non_linear,
        )
        
    def forward(self, x):
        B, N, C = x.shape
        
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q, k, v = self.global_qkv(x)
        
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.global_attn(q=q, k=k, temperature=self.scale)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# %%
# TODO: [1/2] implement into xcit XCA class (XCiT cross-covariance attention)
class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self,
                dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                masks=None,
                mask_levels=3,
                global_heads=3,
                global_proj_type='mix',
                non_linear=False,
                ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(global_heads, 1, 1))

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.global_qkv = Global_QKV(
            dim=dim,
            local_heads=num_heads,
            global_heads=global_heads,
            qkv_bias=qkv_bias
        )
        self.global_attn = Global_Attn(
            masks=masks,
            mask_levels=mask_levels,
            local_heads=num_heads,
            global_heads=global_heads,
            global_proj_type=global_proj_type,
            non_linear=non_linear,
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q, k, v = self.global_qkv(x)
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # ATTN
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.global_attn(q=q, k=k, temperature=self.temperature)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


# %%
if __name__ == '__main__':
    _mask_levels = 3
    _local_heads = 8
    _global_heads = 2
    x_shape = [7, 197, 64]
    # B=7, N=197, H=8, D=8
    
    # masks_index = np.random.randint(0, _mask_levels, [1, 1, x_shape[1], x_shape[1], 1])
    xca_head_d = int(x_shape[2] // _local_heads)
    masks_index = np.random.randint(0, _mask_levels, [1, 1, xca_head_d, xca_head_d, 1])
    
    masks_np = masks_index == np.arange(_mask_levels)
    masks = torch.tensor(masks_np, dtype=torch.float32, device='cuda', requires_grad=False,)
    
    _layer = XCA(
        dim=x_shape[2],
        num_heads=_local_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        
        masks=masks,
        mask_levels=_mask_levels,
        global_heads=_global_heads,
        global_proj_type='mix',
        non_linear=True,
    )
    _layer.to('cuda')
    _input = torch.ones(x_shape, device='cuda')
    _output = _layer(_input)
    
    print('input:', _input.shape)
    print('output:', _output.shape)
    print('input:', _input.shape)
    print('[DONE] XCA')

# %%