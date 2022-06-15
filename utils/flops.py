# %%
import numpy as np
import pandas as pd
import time, os, re, string, json
import plotly.express as px
import plotly.graph_objects as go

# %%
base_values = {
    # 'dim': 192,
    # 'local_heads': 6,
    # 'global_heads': 1,
    # 'mask_levels': 2,
    # 'token_count': 7**2,
    # 'bs': 256,
    
    'dim': 384,
    'local_heads': 6,
    'global_heads': 1,
    'mask_levels': 3,
    'token_count': 14**2,
    'bs': 256,
}

base_values

# def calc_gflops(n, c, h, g, l=3, b=256, **kwargs):
def calc_gflops(
            dim,
            local_heads,
            global_heads,
            mask_levels,
            token_count,
            mix=True,
            bs=256,
            **kwargs):
    flops_baseline = token_count * dim * (3 * dim + 2 * token_count)
    if mix:
        flops_fishpp_mix = token_count * token_count * global_heads * local_heads * (2 + mask_levels/bs)
    else:
        flops_fishpp_mix = 0.
    flops_fishpp = token_count * ( dim * dim * (2 * global_heads/local_heads + 1) + token_count * dim * (global_heads/local_heads + 1) ) + flops_fishpp_mix
    return {
        'baseline': flops_baseline / 1e9,
        'fishpp': flops_fishpp / 1e9,
        'mix': flops_fishpp_mix / 1e9,
        'ratio': flops_fishpp / flops_baseline,
    }

calc_gflops(**base_values)

# %%
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

# %%
class Attn_Baseline(nn.Module):
    def __init__(self,
                dim=128,
                local_heads=8,
                **kwargs,
                ):
        super().__init__()
        self.local_heads = local_heads
        head_dim = dim // local_heads
        self.head_dim = head_dim
        
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x):
        B, N, C = x.shape
        q = self.proj_q(x).reshape(B, N, self.local_heads, -1).permute(0, 2, 1, 3)
        k = self.proj_k(x).reshape(B, N, self.local_heads, -1).permute(0, 2, 1, 3)
        v = self.proj_v(x).reshape(B, N, self.local_heads, -1).permute(0, 2, 1, 3)
        
        # [B, GH, N, N]
        # attn = torch.einsum('bgnc,bgmc->bgnm', q, k)
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return x

# current implementation
class Attn(nn.Module):
    def __init__(self,
                dim=128,
                local_heads=8,
                global_heads=4,
                mask_levels=3,
                masks=None,
                **kwargs,
                ):
        super().__init__()
        self.local_heads = local_heads
        self.global_heads = global_heads
        head_dim = dim // local_heads
        self.head_dim = head_dim
        self.global_dim = head_dim * global_heads
        self.mask_levels = mask_levels
        
        self.proj_q = nn.Linear(dim, self.global_dim, bias=False)
        self.proj_k = nn.Linear(dim, self.global_dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        
        # [N, N, L]
        self.masks = masks
        # [L, GH*H] (mix)
        self.mask_proj = nn.Parameter(torch.ones(self.mask_levels, global_heads * local_heads))
    
    def forward(self, x):
        B, N, C = x.shape
        q = self.proj_q(x).reshape(B, N, self.global_heads, -1).permute(0, 2, 1, 3)
        k = self.proj_k(x).reshape(B, N, self.global_heads, -1).permute(0, 2, 1, 3)
        v = self.proj_v(x).reshape(B, N, self.local_heads, -1).permute(0, 2, 1, 3)
        
        # [B, GH, N, N]
        # attn = torch.einsum('bgnc,bgmc->bgnm', q, k)
        attn = (q @ k.transpose(-2, -1))
        
        mask_weights = self.masks @ self.mask_proj
        
        mask_weights = mask_weights.reshape([1, N, N, self.global_heads, self.local_heads])
        # attn [B, N, N, GH, 1] x mask_weights [1, N, N, GH, H] -> [B, N, N, GH, H]
        attn = attn.permute(0, 2, 3, 1).unsqueeze(-1) * mask_weights
        # [B, N, N, GH, H] -> [B, N, N, H]
        attn = torch.sum(attn, dim=-2, keepdim=False)
        
        attn = attn.permute(0, 3, 1, 2).contiguous()
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return x


# %%
# _bs = 256
# _n = int(14 ** 2)
# _dim = 512
# _levels = 2
# _heads = 16
# _global_heads = 5

def get_feed(token_count, mask_levels, bs, dim, device=None, **kwargs):
    a = np.stack(np.meshgrid(np.arange(token_count), np.arange(token_count)), -1)
    _mask = ((a < (token_count / 2)).sum(-1) == 1).astype(int)

    _masks = _mask[..., None] == np.arange(mask_levels)
    _mask, _masks.astype(int)
    
    mask = torch.tensor(
        _mask,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )
    masks = torch.tensor(
        _masks,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    masks.shape    # [N, N, L]
    _input = torch.tensor(
        np.random.sample([bs, token_count, dim]),
        # np.random.sample([_bs, _global_heads, _n, _dim // _heads]),
        dtype=torch.float32,
        device=device,
    )
    return _input, mask, masks

_input, mask, masks = get_feed(**base_values)
_input.shape, masks.shape

# %%
# model = Attn(
model = Attn_Baseline(
    # dim=_dim,
    # local_heads=_heads, 
    # global_heads=_global_heads,
    # mask_levels=_levels,
    **base_values,
    masks=masks,
)

_flops = float(FlopCountAnalysis(model, _input).total())
_flops / 1e9 / base_values['bs']

# %%
data = []
for mult in [1, 2, 3, 4, 6, 8, 12, 16]:
    for k in [
            'dim',
            'local_heads',
            'global_heads',
            'mask_levels',
            'token_count',
            ]:
        _mod_value = int(base_values[k] * mult)
        _kwargs = {
            **base_values,
            k: _mod_value,
        }
        
        data.append({
            'key': k,
            'mult': mult,
            'mod_value': _mod_value,
            **_kwargs,
            **calc_gflops(**_kwargs),
        })

df = pd.DataFrame(data)
df

fig = px.line(
    df,
    x='mult',
    y='ratio',
    color='key',
    hover_data=['mod_value'],
    log_x=True,
    template='plotly_dark',
)
fig

# %%
fig.write_image('./fishpp_deit_flops_scaling.png')

# %%
# df2 = []
# for d in df.to_dict('records'):
#     k = d['key']











# %% fishpp flops comp.
# deit_small_224 patch[16] head[6] dim[384]

r = calc_gflops(
    dim=384,
    local_heads=6,
    token_count=14**2+1,
    bs=256,
    
    global_heads=3,
    mask_levels=3,
    # global_heads=2,
    # mask_levels=4,
    
    mix=False
)
r['fishpp']

# %%
r = calc_gflops(
    dim=192,
    local_heads=3,
    token_count=14**2+1,
    bs=256,
    
    global_heads=3,
    mask_levels=3,
    # global_heads=2,
    # mask_levels=4,
    
    mix=False
)
r

# %%
df_deit = pd.read_csv('./fishpp_deit_eff.csv')
df_deit['flops_ratio'] = df_deit['gflops'] / 4.608
df_deit['flops_block_ratio'] = df_deit['gflops_block'] / 0.116952
df_deit

# %%
# fig = px.line(
fig = px.scatter(
    df_deit.sort_values('acc'),
    # x='gflops',
    
    # x='flops_ratio',
    # hover_data=['type', 'gflops'],
    x='flops_block_ratio',
    hover_data=['type', 'gflops_block'],
    
    y='acc',
    color='fishpp',
    # markers=True,
    color_continuous_scale=[
        [0, 'rgb(255,120,20)'], 
        # [0.5, 'rgb(57, 162, 225)'],
        [1, 'rgb(20,255,220)'],
    ],
    size=[12]*df_deit.shape[0],
)
fig.update_layout(
    margin={k: 6 for k in 'tlbr'},
    template='plotly_dark',
    # x_range=(0, max(df_deit['gflops'])),
    xaxis=dict(
        # range=[0.9, 1.02],
    ),
    yaxis=dict(
        range=[78.8, 79.6],
    ),
    # showlegend=False,
    height=600,
    width=800,
).update_coloraxes(
    showscale=False,
).show()

# %%
# fig.write_image('./fishpp_deit_flops.png')
fig.write_image('./fishpp_deit_flops_block.png')

# %%
r = calc_gflops(
    dim=96,
    local_heads=12,
    token_count=7**2,
    bs=256,
    
    global_heads=4,
    mask_levels=3,
    # global_heads=2,
    # mask_levels=4,
    
    mix=True,
)
r
