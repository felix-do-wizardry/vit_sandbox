# %% requirements
# python -m pip install -U plotly kaleido nbformat pandas numpy

# %%
# from matplotlib.pyplot import tick_params
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
            # 'tick': 48,
            # 'tick': 60,
            'tick': 72,
            
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
        
        fig.update_traces(
            line=dict(width=self.size['line']),
            marker=dict(size=self.size['marker'], symbol='square'),
        )
        # if with_line:
        #     fig.update_traces(
        #         line=dict(width=self.size['line']),
        #     )
        # if with_marker:
        #     fig.update_traces(
        #         marker=dict(size=self.size['marker'], symbol='square'),
        #     )
                
        
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
            x_title='',
            y_title='',
            legend_title='',
            # corner='tr',
            # axis_angle=90,
            showlegend=False,
            **{
                k: v
                for k, v in kwargs.items()
                if k in [
                    'x_dtick', 'y_dtick',
                    'axis_angle',
                    'with_line', 'with_marker',
                ]
            },
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


FF = FigFormat(fig_size=[800, 800],)
FF

# %%
# # %%
# df = pd.read_csv('efficiency/MEMORY_FLOPS.csv')
# df_block = pd.read_csv('efficiency/MEMORY_FLOPS_BLOCK.csv')
# df['seq_len_str'] = [str(v) for v in df['seq_len']]
# df_block['seq_len_str'] = [str(v) for v in df_block['seq_len']]
# df
# df_block

# # %%
# df = df[df['type_full'] == 'fiak_0.5_0.0'].copy(True)
# df_block = df_block[df_block['type_full'] == 'fiak_0.5_0.0'].copy(True)

# df_block

# # %% MODEL
# fig_flops = px.line(
#     df,
#     x='seq_len_str',
#     y='flop',
#     color='type_full',
#     markers=True,
# )
# figs_flops = FF.format_dual(
#     fig_flops,
#     y_title='FLOPS Ratio',
#     y_dtick=0.05,
# )
# figs_flops[0].show(), figs_flops[1].show()

# # %%
# fig_mem = px.line(
#     df,
#     x='seq_len_str',
#     y='mem',
#     color='type_full',
#     markers=True,
# )
# figs_mem = FF.format_dual(
#     fig_mem,
#     y_title='Memory Ratio',
#     y_dtick=0.1,
# )
# figs_mem[0].show(), figs_mem[1].show()

# # %%



# # %% BLOCK
# fig_flops_block = px.line(
#     df_block,
#     x='seq_len_str',
#     y='flop',
#     color='type_full',
#     markers=True,
# )
# figs_flops_block = FF.format_dual(
#     fig_flops_block,
#     y_title='FLOPS Ratio',
#     y_dtick=0.05,
# )
# figs_flops_block[0].show(), figs_flops_block[1].show()

# # %%
# fig_mem_block = px.line(
#     df_block,
#     x='seq_len_str',
#     y='mem',
#     color='type_full',
#     markers=True,
# )
# figs_mem_block = FF.format_dual(
#     fig_mem_block,
#     y_title='Memory Ratio',
#     y_dtick=0.1,
# )
# figs_mem_block[0].show(), figs_mem_block[1].show()


# %%
# FigFormat.save_plots_md(
#     dp='plots/eff_lm',
#     begin_lines=[
#         '# Plots and stuff',
#         '## Plots LM Ratio',
#     ],
#     image_width=320,
    
#     lm_ratio_flops={
#         'clean': figs_flops[0],
#         'ref': figs_flops[1],
#     },
#     lm_ratio_mem={
#         'clean': figs_mem[0],
#         'ref': figs_mem[1],
#     },
    
#     lm_ratio_flops_block={
#         'clean': figs_flops_block[0],
#         'ref': figs_flops_block[1],
#     },
#     lm_ratio_mem_block={
#         'clean': figs_mem_block[0],
#         'ref': figs_mem_block[1],
#     },
# )

# %%
df_fish_lm_test = pd.read_csv('efficiency/lm_fish/mem_time_test_raw.csv')
df_fish_lm_train = pd.read_csv('efficiency/lm_fish/mem_time_train_raw.csv')
df_fish_lm_test['seq_len_str'] = [str(v) for v in df_fish_lm_test['sequence_len']]
df_fish_lm_train['seq_len_str'] = [str(v) for v in df_fish_lm_train['sequence_len']]
df_fish_lm_test
df_fish_lm_train

# %%
cols = ['name_method', 'num_global_heads', 'd_model', 'num_local_heads', 'd_head', 'sequence_len',]
{
    col: np.mean(df_fish_lm_test[col] == df_fish_lm_train[col])
    for col in cols
}

# %%
df_fish_lm = pd.read_csv('efficiency/lm_fish/model_raw.csv')
df_fish_lm['seq_len_str'] = [str(v) for v in df_fish_lm['sequence_len']]
df_fish_lm

# %%
df_fish_lm_softmax = df_fish_lm[df_fish_lm['name_method'] == 'softmax'].copy(True)
df_fish_lm_test_softmax = df_fish_lm_test[df_fish_lm_test['name_method'] == 'softmax'].copy(True)
df_fish_lm_train_softmax = df_fish_lm_train[df_fish_lm_train['name_method'] == 'softmax'].copy(True)
df_fish_lm_softmax

# %% LM MODEL FLOPS
fig_flops = px.line(
    df_fish_lm_softmax,
    x='seq_len_str',
    y='flops',
    color='d_model',
    markers=True,
)
# fig_flops.show()
figs_flops = tuple([
    FF.format(
        go.Figure(fig_flops),
        x_title='',
        y_title='',
        legend_title='',
        corner='tl',
        showlegend=False,
    ),
    FF.format(
        go.Figure(fig_flops),
        x_title='Sequence Length',
        y_title='FLOPS',
        legend_title='Model Dim',
        corner='tl',
        showlegend=True,
    ),
])
figs_flops[0].show()
figs_flops[1].show()

# %% LM MODEL MEM TEST
fig_mem_test = px.line(
    df_fish_lm_test_softmax.sort_values('d_model'),
    y='test_memory',
    # x='d_model',
    x='seq_len_str',
    color='d_model',
    markers=True,
)
# fig_mem.show()
figs_mem_test = tuple([
    FF.format(
        go.Figure(fig_mem_test),
        x_title='',
        y_title='',
        legend_title='',
        corner='tl',
        showlegend=False,
    ),
    FF.format(
        go.Figure(fig_mem_test),
        x_title='Sequence Length',
        y_title='Memory Test',
        legend_title='Model Dim',
        corner='tl',
        showlegend=True,
    ),
])
figs_mem_test[0].show()
figs_mem_test[1].show()

# %% LM MODEL MEM TRAIN
fig_mem_train = px.line(
    df_fish_lm_train_softmax.sort_values('d_model'),
    y='train_memory',
    # x='d_model',
    # color='sequence_len',
    x='seq_len_str',
    color='d_model',
    markers=True,
)
# fig_mem.show()
figs_mem_train = tuple([
    FF.format(
        go.Figure(fig_mem_train),
        x_title='',
        y_title='',
        legend_title='',
        corner='tl',
        showlegend=False,
    ),
    FF.format(
        go.Figure(fig_mem_train),
        x_title='Sequence Length',
        y_title='Memory Train',
        legend_title='Model Dim',
        corner='tl',
        showlegend=True,
    ),
])
figs_mem_train[0].show()
figs_mem_train[1].show()



# %%
FigFormat.save_plots_md(
    dp='plots/fish_lm_eff',
    begin_lines=[
        '# Plots and stuff',
        '## Plots LM Ratio',
    ],
    image_width=320,
    
    fish_lm_flops={
        'clean': figs_flops[0],
        'ref': figs_flops[1],
    },
    fish_lm_mem_test={
        'clean': figs_mem_test[0],
        'ref': figs_mem_test[1],
    },
    fish_lm_mem_train={
        'clean': figs_mem_train[0],
        'ref': figs_mem_train[1],
    },
)











# %%
import sys
sys.exit()


# %%
figs_all = {
    k: {
        k2: _figs[k2][k]
        for k2 in ['clean', 'ref']
    }
    for _figs in [figs_ratio_len]
    for i, k in enumerate(_figs['clean'].keys())
    # for k, _fig in _figs['clean'].items()
}
len(figs_all)

# %%
for k, _figs in figs_all.items():
    print(k)
    _figs['ref'].show()

# %%
_dp = './paper_gmm/plots_deit'
_dp = './paper_gmm/plots_swin'

_dp_clean = os.path.join(_dp, 'clean')
_dp_ref = os.path.join(_dp, 'ref')
if not os.path.isdir(_dp_clean):
    os.makedirs(_dp_clean)
if not os.path.isdir(_dp_ref):
    os.makedirs(_dp_ref)

for name, figs in figs_all.items():
    for k2, ___dp in zip(['clean', 'ref'], [_dp_clean, _dp_ref]):
        figs[k2].write_image(os.path.join(___dp, f'{name}.png'))

# for name, fig in figs_all_ref.items():
#     fig.write_image(os.path.join(_dp_ref, f'{name}.png'))

print(f'[PLOT] {len(figs_all)} plots saved in <{_dp}>')

# %%
figs_groups = [
    [figs_ratio_len],
    # [figs_abs_len, figs_abs_dim],
    # [figs_head_len, figs_head_dim],
]
fp_rm = os.path.join(_dp, f'README.md')
readme_lines = [
    '# Model Metrics Plot for gmm deit',
]
image_width = 320
image_cols = 4
readme_lines.append('## PLOTS')
for gi, (name, _figs) in enumerate(figs_all.items()):
    # readme_lines.append([
    #     '## I - TEST BLOCK RATIO',
    #     '## II - TEST MODEL RATIO',
    #     # '## III - RATIO PER HEAD',
    # ][gi])
    
    name_lines = []
    name_lines.append(f'> {name}')
    readme_lines.extend([
        *name_lines,
        '<p float="left" align="left">',
        f'<img src="clean/{name}.png" width="{image_width}" />',
        f'<img src="ref/{name}.png" width="{image_width}" />',
        '</p>',
    ])

# readme_lines

# %%
readme_txt = '\n\n'.join(readme_lines)

with open(fp_rm, 'w') as fo:
    fo.writelines(readme_txt)
print(f'[PLOT] saved README.md at <{fp_rm}>')

# %%