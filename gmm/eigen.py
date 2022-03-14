# %%
import numpy as np
import json
import os, sys, time

import numpy as np
import pandas as pd
import json, time, re, string, os

import plotly.express as px
import plotly.graph_objects as go

# %%
class FigFormat:
    def __init__(self, fig_size=[800, 800]):
        self.fig_size = fig_size
        
        self.font = {
            # 'axis': 32,
            # 'tick': 30,
            # 'legend': 16,
            
            # title font size
            'axis': 48,
            # 'axis': 72,
            
            # text number font size
            'tick': 48,
            # 'tick': 72,
            
            'legend': 20,
            # 'legend': 48,
            # 'legend': 96,
            
        }
        self.size = {
            'line': 4,
            'marker': 12,
            
            'linewidth': 5,
            'tickwidth': 3,
            'gridwidth': 1,
        }
        self.fig_config = {
            # 'legend_opacity': 0.5,
            'legend_opacity': 1.0,
            'legend_color_value': 255,
        }
    
    def format(self,
                fig,
                x_title='Sequence Length',
                y_title='Ratio',
                legend_title='Type',
                corner='tr',
                x_dtick=None,
                y_dtick=None,
                axis_angle=90,
                showlegend=True,
                with_line=True,
                with_marker=True,
                **kwargs,
                ):
        
        fig.update_layout(
            font={
                'color': '#000000',
                'family': 'Helvetica',
            },
            paper_bgcolor="#FFFFFF",
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                font=dict(color='#000000'),
            ),
        )

        axes = dict(
            color='#000000',
            # showgrid=False,
            linecolor='#000000',
            gridcolor='#aaaaaa',
            tickcolor='#000000',
            mirror=True,
            
            linewidth=self.size['linewidth'],
            tickwidth=self.size['tickwidth'],
            gridwidth=self.size['gridwidth'],
            
            title_font = {"size": self.font['axis'],},
            # title_standoff = 16,
            tickfont={'size': self.font['tick'],},
            ticks='outside',
            
            # dtick=0.02,
        )
        fig.update_xaxes(
            **axes,
        )
        fig.update_yaxes(
            **axes,
        )
        
        if with_line:
            fig.update_traces(
                line=dict(width=self.size['line']),
            )
        if with_marker:
            fig.update_traces(
                marker=dict(size=self.size['marker'], symbol='square'),
            )
                
        
        legend_margin = 0.02
        legend_pos_dict = {
            f'{_yk[0]}{_xk[0]}': dict(
                yanchor=_yk,
                xanchor=_xk,
                x=legend_margin + _x * (1 - legend_margin * 2),
                y=legend_margin + _y * (1 - legend_margin * 2),
            )
            for _x in [0, 1]
            for _y in [0, 1]
            for _xk in [['left', 'right'][_x]]
            for _yk in [['bottom', 'top'][_y]]
        }
        assert corner in legend_pos_dict
        _bgcolor = 'rgba({0}, {0}, {0}, {1})'.format(
            self.fig_config['legend_color_value'],
            self.fig_config['legend_opacity'],
        )
        fig.update_layout(
            width=self.fig_size[0],
            height=self.fig_size[1],
            xaxis=dict(
                title_text=x_title,
                tickangle=axis_angle,
                # dtick=None if x_dtick is None else x_dtick,
            ),
            yaxis=dict(
                title_text=y_title,
            ),
            autosize=False,
            showlegend=showlegend,
            legend=dict(
                title_text=legend_title,
                
                **legend_pos_dict[corner],
                font=dict(size=self.font['legend'],),
                # bgcolor='rgba(255, 255, 255, 0.75)',
                bgcolor=_bgcolor,
                # bordercolor="rgba(0, 0, 0, 0.25)",
                # borderwidth=3,
            ),
            **kwargs,
        )
        if x_dtick is not None:
            fig.update_xaxes(
                dtick=x_dtick,
            )
        if y_dtick is not None:
            fig.update_yaxes(
                dtick=y_dtick,
            )
        
        return fig
    
    def format_dual(self, fig, **kwargs):
        fig_clean = go.Figure(fig)
        fig_ref = go.Figure(fig)
        fig_clean = self.format(
            fig_clean,
            # corner='tr',
            # axis_angle=90,
            **{
                **kwargs,
                'x_title': '',
                'y_title': '',
                'legend_title': '',
                'showlegend': False,
            },
            # **{
            #     k: v
            #     for k, v in kwargs.items()
            #     if k in [
            #         'x_dtick', 'y_dtick',
            #         'axis_angle',
            #         'with_line', 'with_marker',
            #         'bargap',
            #     ]
            # },
        )
        fig_ref = self.format(
            fig_ref,
            **kwargs,
        )
        return fig_clean, fig_ref
    
    @classmethod
    def save_plots_md(cls,
                dp='./plots',
                begin_lines=[],
                image_width=400,
                **kwargs,
                ):
        # kwargs is a dict of dict of figs
        fp_rm = os.path.join(dp, f'README.md')
        readme_lines = [
            # '# Model Metrics Plot for gmm deit',
            # '## PLOTS',
            *begin_lines,
        ]
        # image_width = 320
        # image_cols = 4
        
        for name, figd in kwargs.items():
            _lines = [
                f'> {name}',
                '<p float="left" align="left">',
            ]
            for _ver, _fig in figd.items():
                _fp_rel = f'{_ver}/{name}.png'
                _fp = os.path.join(dp, _fp_rel)
                _dp_ver = os.path.join(dp, _ver)
                if not os.path.isdir(_dp_ver):
                    os.makedirs(_dp_ver)
                
                _fig.write_image(_fp)
                
                _lines.extend([
                    f'<img src="{_fp_rel}" width="{image_width}" />',
                    # f'<img src="ref/{name}.png" width="{image_width}" />',
                ])
            _lines.extend([
                '</p>',
            ])
            readme_lines.extend(_lines)
        
        readme_txt = '\n\n'.join(readme_lines)

        with open(fp_rm, 'w') as fo:
            fo.writelines(readme_txt)
        print(f'[PLOT] saved with README.md in <{dp}>')



FF = FigFormat(
    fig_size=[800, 800],
)
FF

# %%
def format_fig_dual_eigen(fig, show=False, **kwargs):
    fig_clean, fig_ref = FF.format_dual(
        fig,
        **kwargs,
    )
    _layout = dict(
        xaxis = dict(
            type="log",
            tickmode = 'array',
            tickvals = [(10 ** i) for i in range(-10, 11)],
            ticktext = [f'10<sup>{i}</sup>' for i in range(-10, 11)],
        )
    )
    # fig_clean.update_xaxes(type="log")
    # fig_ref.update_xaxes(type="log")
    fig_clean.update_layout(**_layout)
    fig_ref.update_layout(**_layout)
    if show:
        fig_clean.show()
        fig_ref.show()
    return fig_clean, fig_ref


# %%
dp = 'eigen'
names = [
    # 'baseline',
    'gmm_gd_0.65',
    'gmm_gd_0.7_0.15',
]
eigen_datas = np.stack([
    np.load(os.path.join(dp, f'{v}_W_all.npy'))
    for v in names
])

# [Z, L, N^2]
eigen_datas.shape

# %%
a = np.clip(eigen_datas[0, 0], 0, None)
a[:5], a[-5:]

# %%
thr = 0.95
a_ene = np.zeros_like(eigen_datas, dtype=float)
time_start = time.time()
# for i in list(range(len(a))):
for z in range(eigen_datas.shape[0]):
    for l in range(eigen_datas.shape[1]):
        a_sum = np.sum(eigen_datas[z, l])
        for i in range(eigen_datas.shape[2]):
            r = np.sum(eigen_datas[z, l, -1 - i : ]) / a_sum
            a_ene[z, l, i] = r
            # if r >= 0.95:
            #     print(i, r)
            #     break
# a_ene = np.array(a_ene)
a_ene.shape


# %%
# _limit = a_ene.shape[-1]
_limit = 1200
df_ene = pd.DataFrame([
    {
        'model_index': z,
        'model': names[z],
        'layer': l,
        'index': i,
        'energy': a_ene[z, l, i],
    }
    for z in range(a_ene.shape[0])
    for l in range(a_ene.shape[1])
    for i in range(min(a_ene.shape[2], _limit))
])
df_ene

# %%
a_ene_model = np.mean(a_ene, axis=1)
df_ene_model = pd.DataFrame([
    {
        'model_index': z,
        'model': names[z],
        'index': i,
        'energy': a_ene_model[z, i],
    }
    for z in range(a_ene_model.shape[0])
    for i in range(min(a_ene_model.shape[1], _limit))
])
df_ene_model

# %% MODEL
fig_model = px.line(
    df_ene_model,
    x='index',
    y='energy',
    color='model',
    # log_x=True,
)
figs_model = format_fig_dual_eigen(
    fig_model,
    1,
    axis_angle=0,
    y_title='Cumulative Eigen Energy',
    x_title='# Eigen Values',
    legend_title='Model',
)

# %% 
_model_index = 0
layers = [0, 1, 9, 10]
_mask = np.all([
    df_ene['model_index'] == _model_index,
    np.sum([
        df_ene['layer'] == _layer
        for _layer in layers
    ], axis=0),
], axis=0)
print(names[_model_index])

_df = df_ene[_mask]
fig_layer = px.line(
    _df,
    x='index',
    y='energy',
    color='layer',
    hover_data=['model'],
    # y=a_ene[:_limit],
    # x=np.arange(_limit) + 1,
    # log_x=True,
)
fig_layer
figs_layer = format_fig_dual_eigen(
    fig_layer,
    1,
    axis_angle=0,
    y_title='Cumulative Eigen Energy',
    x_title='# Eigen Values',
    legend_title='Layer',
)


# %%
_model_index = 0
_limit2 = 50
# layers = [0, 1, 9, 10]
eigen_mean = np.mean(eigen_datas[_model_index, :, ::-1][:, :_limit2], axis=0)
eigen_mean.shape

fig_val = px.bar(
    y=eigen_mean,
    x=np.arange(eigen_mean.shape[0]) + 1,
)

figs_val = format_fig_dual_eigen(
    fig_val,
    1,
    axis_angle=0,
    y_title='Eigen Value',
    x_title='Eigen Values #',
    # legend_title='Layer',
    with_line=False,
    with_marker=False,
    bargap=0.,
)



# %%

# %%
FigFormat.save_plots_md(
    dp='plots/eigen_deit',
    begin_lines=[
        '# Plots and stuff',
        '## Plots deit eigen values',
    ],
    image_width=320,
    
    deit_eigen_model={
        'clean': figs_model[0],
        'ref': figs_model[1],
    },
    
    deit_eigen_layer_65={
        'clean': figs_layer[0],
        'ref': figs_layer[1],
    },
    
    deit_eigen_val={
        'clean': figs_val[0],
        'ref': figs_val[1],
    },
)



# %%
sys.exit()

# %%
import numpy as np
import json
import time

# from scipy.linalg import eigh
import torch
eigh = torch.linalg.eigh
eigvals = torch.linalg.eigvals

# %%
# A = cov_matrices[i]
# W, V = eigh(A)
# np.save('/tanData/cov_matrices/'+name+'_100k_W'+str(i),W)
# np.save('/tanData/cov_matrices/'+name+'_100k_V'+str(i),V)

# %%
# t = 8
# n = t * t + 1
# a = np.random.sample([n, n])
# a.shape

# %%
def calc_eigh(attn):
    time_start = time.time()
    
    # a = torch.tensor(attn, device='cuda')
    a = attn
    _b = a.reshape(-1, 1) * a.reshape(1, -1)
    
    time_mult = time.time() - time_start
    
    time_start = time.time()
    b = torch.tensor(_b, device='cuda')
    W, V = eigh(b)
    W, V = eigh(b)
    
    time_eigh = time.time() - time_start
    time_total = time_mult + time_eigh
    
    del W
    del V
    del b
    del _b
    return time_mult, time_eigh, time_total
    # return W, V, time_mult, time_eigh, time_total

# %%
# for n in [145]:
#     t = round(np.sqrt(n-1), 2)
# for i, t in enumerate(range(10, 13)):
for i, t in enumerate(range(4, 9)):
    if i > 0: time.sleep(4)
    
    n = t * t + 1
    attn = np.random.sample([n, n])
    
    # W, V, time_mult, time_eigh, time_total = calc_eigh(attn)
    
    torch.cuda.empty_cache()
    time_start = time.time()
    _ = calc_eigh(attn)
    time_total = time.time() - time_start
    
    # mem = torch.cuda.max_memory_allocated()
    # print(f't[{t}] n[{n}] attn[{attn.shape}] b[{b.shape}] W[{W.shape}] V[{V.shape}]')
    # print(f'time: total[{time_total:.2f}s] mult[{time_mult:.2f}s] eigh[{time_eigh:.2f}s]')
    print()
    print(f't[{t}] n[{n}] attn[{list(attn.shape)}] time_total[{time_total:.2f}s]')
    del attn

# %%
import sys
sys.exit()

# %%
import pandas as pd
import plotly.express as px

# %%
df = pd.read_csv('/home/hai/vit_sandbox/gmm/eigen_log.csv')
df

# %%
fig = px.line(
    df,
    x='t',
    # y=['time', 'mem'],
    y='time',
    template='plotly_dark',
    height=320,
    log_y=True,
)
fig.update_layout(margin=dict(t=0,b=0,l=0,r=0)).show()
fig = px.line(
    df,
    x='t',
    y='mem',
    template='plotly_dark',
    height=320,
    log_y=True,
)
fig.update_layout(margin=dict(t=0,b=0,l=0,r=0)).show()
