# %%
import wandb
import numpy as np
import pandas as pd
import time, os, json, re, string

import plotly.express as px
import plotly.graph_objects as go

# %%
api = wandb.Api()
runs = api.runs("fpt-team/fiak_deit_metrics")

# %%
summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list,
})

runs_df

# %%
df = pd.DataFrame([
    {
        'name': d['name'],
        **{
            k: d['summary'].get(k, 0)
            # for k, v in d['summary'].items()
            # if not k.startswith('_')
            for k in [
                'bs',
                'image_size', 'patch_size',
                'N',
                'type',
                'mem_gb_train', 'mem_gb_train_test',
                'gflops',
                'speed_train', 'speed_test',
                'mode',
                'prune_total', 'prune_k',
            ]
        },
    }
    for d in runs_df.to_dict('records')
])

df['patch_size'] = [v if v > 0 else 4 for v in df['patch_size']]
df['N'] = [v if v > 0 else 197 for v in df['N']]

prune_totals = []
prune_ks = []
attn_types = []
for i, d in enumerate(df.to_dict('records')):
    
    _type = d['type']
    _type_comps = _type.split('_')
    _prune_total = d['prune_total']
    _prune_k = d['prune_k']
    _attn_type = _type_comps[0]
    
    if len(_type_comps) >= 3:
        if _prune_total <= 0:
            _prune_total = float(_type_comps[1])
        if _prune_k <= 0:
            _prune_k = float(_type_comps[2])
    
    prune_totals.append(_prune_total)
    prune_ks.append(_prune_k)
    attn_types.append(_attn_type)

df['prune_total'] = prune_totals
df['prune_k'] = prune_ks
df['attn_type'] = attn_types
df

# %%
df.to_csv('data/raw_C.csv')


# %%
def fig_clean(fig, size=[800, 640], margin=2):
    fig.update_layout(
        margin={k: margin for k in 'tblr'},
        width=size[0],
        height=size[1],
        template='plotly_dark',
    )
    return fig

def img_matrix(img, sep=1, fill=None):
    # 4D
    assert len(img.shape) == 4
    assert isinstance(sep, int) and sep >= 0
    mh, mw, ch, cw = img.shape
    h = mh * ch + (mh - 1) * sep
    w = mw * cw + (mw - 1) * sep
    imgm = np.full(
        shape=[h, w],
        fill_value=fill,
        dtype=np.float32,
    )
    for my in range(mh):
        for mx in range(mw):
            imgm[
                my * (ch + sep): my * (ch + sep) + ch,
                mx * (cw + sep): mx * (cw + sep) + cw
            ] = img[my, mx]
    
    return imgm

# %%
# df2 = df[:13].copy()
df2 = df[
    np.any([
        df['N'] == 785,
        df['N'] == 197,
        df['mode'] != 'train',
    ], axis=0)
].copy().reset_index(drop=True)
# df2
# df2 = df2[df2['mode'] != 'train'].copy()
df2
# df2['run'] = [f"{df2['type'][i]} - {df2['mode'][i]}" for i in range(df2.shape[0])]
df2['run'] = [
    f"{df2['type'][i]} N[{df2['N'][i]}]"
    for i in range(df2.shape[0])
]
df2


# %%
fig = fig_clean(px.scatter(
    df2.sort_values(['N', 'mode', 'type']),
    x='run',
    # x='type',
    # x='attn_type',
    color='mode',
    # y='speed_train',
    y='speed_test',
    # color=''
    size=[10] * df2.shape[0],
))
fig.show()

# %% INFERENCE
# df2[(df2['type'] == 'base') & (df2['mode'] == 'metric')]
data_ratio = []
for _N in [197, 785]:
    _df = df2[(df2['mode'] != 'train') & (df2['N'] == _N)]
    d_base = df2[(df2['type'] == 'base') & (
        df2['mode'] == 'metric') & (
        (df2['N'] == _N)
        )].to_dict('records')[0]
    _data_ratio = [
        {
            **d,
            # 'speed_test': d['speed_test'] / d_base['speed_test'],
            **{
                k: d[k] / d_base[k]
                for k in ['speed_test', 'gflops', 'mem_gb_train_test']
            },
        }
        for d in _df.to_dict('records')
    ]
    data_ratio.extend(_data_ratio)

df_ratio = pd.DataFrame(data_ratio)
df_ratio

# %% 1
fig = fig_clean(px.line(
    df_ratio.sort_values(['type']),
    x='type',
    # y=['mem_gb_train_test'],
    # y=['gflops'],
    y=['speed_test'],
    
    color='mode',
    line_dash='N',
    # line_dash='mode',
    # color='N',
    
)).update_layout(
    # title="Plot Title",
    # xaxis_title="X Axis Title",
    # yaxis_title='test memory ratio (lower=better)',
    # yaxis_title='test flops ratio (lower=better)',
    yaxis_title='test speed ratio (higher=better)',
    # legend_title="Legend Title",
)
fig.show()
# fig.write_image('plots/')

# %% 2
plot_configs = {
    'mem_gb_train_test': 'test memory ratio (lower=better)',
    'gflops': 'test flops ratio (lower=better)',
    'speed_test': 'test speed ratio (higher=better)',
}
for k, v in plot_configs.items():
    fig = fig_clean(px.line(
        df_ratio.sort_values(['mode']),
        x='mode',
        y=k,
        # y=['mem_gb_train_test'],
        # y=['gflops'],
        # y=['speed_test'],
        
        color='type',
        line_dash='N',
        # line_dash='mode',
        # color='N',
        
    )).update_layout(
        # title="Plot Title",
        # xaxis_title="X Axis Title",
        yaxis_title=v,
        # yaxis_title='test memory ratio (lower=better)',
        # yaxis_title='test flops ratio (lower=better)',
        # yaxis_title='test speed ratio (higher=better)',
        # legend_title="Legend Title",
    )
    fig.show()
    fig.write_image(f'plots/ratio_mode_{k}.png')

# %%
# runs_df.to_csv("project.csv")

# %%