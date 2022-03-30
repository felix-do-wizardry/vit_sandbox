# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json, os, time, datetime

# import wandb
import sys
from fvcore.nn import FlopCountAnalysis

from backbones.vision_transformer import VisionTransformer_FishPP, VisionTransformer

# import warnings
# warnings.filterwarnings("ignore", message='Unsupported operator*')
# # Unsupported operator aten::add encountered 25 time(s)
# # Unsupported operator aten::mul encountered 12 time(s)
# # Unsupported operator aten::softmax encountered 12 time(s)
# # Unsupported operator aten::gelu encountered 12 time(s)

# %%
def get_metrics(model, img_size=224, bs=64):
    torch.cuda.empty_cache()
    model.eval()
    _input = torch.tensor(
        np.ones(shape=[bs, 3, img_size, img_size]) / 2,
        dtype=torch.float32, device='cuda',
    )
    flops_ca = FlopCountAnalysis(model, _input)
    _gflops = float(flops_ca.total() / 1e9)
    _mem_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
    _metrics = {
        'gflops': _gflops / bs,
        'memory': _mem_gb / bs,
    }
    del _input
    torch.cuda.empty_cache()
    return _metrics

def get_model_metrics(fishpp=True, img_size=224, bs=64, **kwargs):
    if fishpp:
        model = VisionTransformer_FishPP(img_size=img_size, **kwargs).cuda()
    else:
        model = VisionTransformer(img_size=img_size, **kwargs).cuda()
    
    m = get_metrics(model, img_size=img_size, bs=bs)
    del model
    return {
        'fishpp': bool(fishpp),
        'img_size': img_size,
        'bs': bs,
        **m,
        # **kwargs,
    }

# %%
global_proj_types = ['full', 'mix']
cls_token_types = ['copy', 'pos', 'mask', 'sum']

_img_size = 224
_bs = 64

base_kwargs = dict(
    in_chans=3,
    num_classes=1000,
    # mlp_ratio=4.,
    # qkv_bias=True,
    # representation_size=None,
    # distilled=False,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.,
    # embed_layer=PatchEmbed,
    # norm_layer=None,
    # act_layer=None,
    # weight_init='',
)

fish_kwargs = dict(
    mask_levels=3,
    global_heads=3,
    mask_type='dist',
    non_linear=0,
    non_linear_bias=1,
    
    global_proj_type='mix',
    cls_token_pos=0.5,
    cls_token_type='sum',
    
    layer_limit=8,
    layer_offset=0,
    
    metrics_test=True,
)

model_kwargs = dict(
    img_size=_img_size,
    patch_size=16,
    depth=12,
    embed_dim=384,
    num_heads=6,
)

# %%
class Combination:
    def __init__(self, **kwargs):
        assert len(kwargs) > 0
        # self.data = {k: None for k, v in kwargs.items()}
        self.count = len(kwargs)
        self.fields = []
        for k, v in kwargs.items():
            if not isinstance(v, list):
                v = [v]
            assert len(v) > 0
            # self.counts.append(len(v))
            self.fields.append({
                'key': k,
                'index': 0,
                'current': v[0],
                'values': v,
                'count': len(v),
            })
    
    def __iter__(self):
        # self.n = 0
        self.starting = True
        return self
    
    def __next__(self):
        for i in list(range(self.count))[::-1]:
            if self.starting:
                self.starting = False
                break
            if self.fields[i]['index'] < self.fields[i]['count'] - 1:
                self.fields[i]['index'] += 1
                break
            else:
                if i == 0:
                    raise StopIteration
                self.fields[i]['index'] = 0
        
        return {
            d['key']: d['values'][d['index']]
            for d in self.fields
        }

list(Combination(a=[1,2,3], b=['x', 'z']))

# %%
metrics = []
# for d in Combination(img_size=[224, 384], bs=[64], patch_size=[16, 8], depth=[12], embed_dim=[384], num_heads=[6]):
for d in Combination(img_size=[384], bs=[1], patch_size=[16, 8], depth=[12], embed_dim=[384], num_heads=[6]):
    print()
    print(d, '\n')
    N = int(int(d['img_size'] / d['patch_size']) ** 2) + 1
    m_base = get_model_metrics(fishpp=False, **{**base_kwargs, **d})
    metrics.append({
        'fishpp': False,
        'N': N,
        **{f'{k}_ratio': 1.0 for k in ['gflops', 'memory']},
        **m_base,
        **d,
    })
    # print(d)
    for _fish_kwargs in Combination(
                global_proj_type=['full', 'mix'],
                # global_proj_type=['full'],
                # cls_token_type=['copy', 'pos', 'mask', 'sum'],
                # cls_token_type=['pos', 'sum'],
                cls_token_type=['sum'],
                mask_levels=[3],
                # non_linear=[0, 1],
                global_heads=[1, 2, 3],
                layer_limit=[8],
                ):
        m = get_model_metrics(fishpp=True, **{
            **base_kwargs,
            **fish_kwargs,
            **_fish_kwargs,
            **d,
        })
        metrics.append({
            'fishpp': True,
            'N': N,
            **{f'{k}_ratio': m[k] / m_base[k] for k in ['gflops', 'memory']},
            **m,
            **d,
            **fish_kwargs,
            **_fish_kwargs,
        })
    

# get_model_metrics(fishpp=True, img_size=224, bs=64, **kwargs)

df = pd.DataFrame(metrics)
print(df)

df.to_csv('metrics/fishpp.csv')

# %%
sys.exit()

# %%

# for _fishpp in [0, 1, 1, 0, 0, 1]:
for _fishpp in [1, 0]:
    if _fishpp:
        model = VisionTransformer_FishPP(
            **base_kwargs,
            **model_kwargs,
            **fish_kwargs,
        ).cuda()
    else:
        model = VisionTransformer(
            **base_kwargs,
            **model_kwargs,
            **fish_kwargs,
        ).cuda()
    
    m = get_metrics(model, img_size=_img_size, bs=_bs)
    del model
    torch.cuda.empty_cache()
    metrics.append({
        'fishpp': bool(_fishpp),
        'bs': 64,
        **m,
        # **,
    })
    
    print(f'\nmetrics: bs[{_bs}]',
        f'fishpp[{_fishpp}]',
        f'fvcore_gflops[{m["gflops"]:.3f}G ({m["gflops"]/_bs:.3f}G)]',
        f'memory[{m["memory"]:.3f}GB ({m["memory"]/_bs:.3f}GB)]\n',)


print('\n')
print(metrics)

# %%
df = pd.DataFrame(metrics)
print(df)

# %%
# _metrics = {
#     'bs': _bs,
#     'gflops': _gflops,
#     'memory': _mem_gb,
#     **dict(
#         fishpp=args.fishpp,
#         global_heads=args.fish_global_heads,
#         mask_type=args.fish_mask_type,
#         mask_levels=args.fish_mask_levels,
#         non_linear=args.fish_non_linear,
#         non_linear_bias=args.fish_non_linear_bias,
#         cls_token_pos=args.fish_cls_token_pos,
        
#         # global_full_proj=args.fish_global_full_proj,
#         # global_full_mix=args.fish_global_full_mix,
#         global_proj_type=args.fish_global_proj_type,
        
#         cls_token_type=args.fish_cls_token_type,
        
#         layer_limit=args.fish_layer_limit,
#         layer_offset=0,
        
        
#         metrics_test=args.metrics_test,
#     ),
# }
# _fp = os.path.join(args.output_dir, 'metrics.json')
# with open(_fp, 'w') as fo:
#     json.dump(_metrics, fo, indent=4)
# print(f'metrics.json: <{_fp}>')
# return


# # %%


#     model = create_model(
#         args.model,
#         pretrained=False,
#         num_classes=args.nb_classes,
#         drop_rate=args.drop,
#         drop_path_rate=args.drop_path,
#         drop_block_rate=None,
        
#         fishpp=args.fishpp,
#         global_heads=args.fish_global_heads,
#         mask_type=args.fish_mask_type,
#         mask_levels=args.fish_mask_levels,
#         non_linear=args.fish_non_linear,
#         non_linear_bias=args.fish_non_linear_bias,
#         cls_token_pos=args.fish_cls_token_pos,
        
#         # global_full_proj=args.fish_global_full_proj,
#         # global_full_mix=args.fish_global_full_mix,
#         global_proj_type=args.fish_global_proj_type,
        
#         cls_token_type=args.fish_cls_token_type,
        
#         layer_limit=args.fish_layer_limit,
#         layer_offset=0,
        
        
#         metrics_test=args.metrics_test,
#     )