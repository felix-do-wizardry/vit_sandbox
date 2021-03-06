# %%
import pickle
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time, os, json

from draft import NumpyImage, ImageReadme

# %%

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
                with_line=False,
                with_marker=False,
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
            
            showgrid=False,
            zeroline=False, # thick line at x=0
            visible=False,  # numbers below
            
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
def read_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

# %%
pi_pkl = read_pkl('pi/pi_img_deit.pkl')
pi_pkl.keys()

# %%
pi = np.array([pi_pkl[0.0][k].squeeze(0) for k in sorted(pi_pkl[0.0].keys())])
pi = np.abs(pi)
pi.shape

# %%
# pi_hists = [
#     np.histogram(
#         bins=100,
#         # a=v.reshape(-1),
#         # range=(b.min(), b.max()),
#         # a=v.reshape(-1),
#         # range=(-0.1, 0.1),
#         a=np.abs(v.reshape(-1)),
#         range=(0, 0.1),
#     )
#     for v in pi[:-1]
# ]
# pi_hist_df = pd.DataFrame([
#     {
#         'layer': i,
#         'value': _value,
#         'count': _count,
#     }
#     for i, (count, values) in enumerate(pi_hists)
#     for _count, _value in zip(count, (values[:-1] + values[1:]) / 2)
# ])
# pi_hist_df

# # %%
# fig = px.line(
#     pi_hist_df,
#     x='value',
#     y='count',
#     color='layer',
# )
# fig.update_layout(
#     template='plotly_dark',
#     yaxis=dict(type='log'),
# )
# fig


# %%
# def index2pos(indices, t=14, cls_count=1):
#     # p = np.zeros(len(indices), dtype=float)
#     p = np.stack([
#         np.floor((indices - cls_count) / t),
#         (indices - cls_count) % t,
#     ], axis=1)
#     indices_cls = np.where(indices < cls_count)[0]
#     p[indices_cls] = [[t/2, t/2] for _ in range(len(indices_cls))]
#     return p

# def token_dist(p0, p1):
#     return np.sqrt(np.sum((p0 - p1) ** 2, axis=-1))

# # %%
# pi_df = pd.DataFrame([
#     {
#         'v': np.abs(v3),
#         'q': _q,
#         'k': _k,
#         # 'dist': token_dist(
#         #     index2pos(_q, 14, 1),
#         #     index2pos(_k, 14, 1),
#         # ),
#         'layer': _layer,
#         'head': _head,
#     }
#     for _layer, v0 in enumerate(pi)
#     for _head, v1 in enumerate(v0)
#     for _q, v2 in enumerate(v1)
#     for _k, v3 in enumerate(v2)
# ])
# pi_df['dist'] = token_dist(
#     index2pos(np.array(pi_df['q']), 14, 1),
#     index2pos(np.array(pi_df['k']), 14, 1),
# ) / (13 * np.sqrt(2))
# pi_df

# %%


# %% dist correlation np

# _layer = 9
# h2d = np.histogram2d(
#     pi_df[pi_df['layer'] == _layer]['dist'],
#     np.abs(pi_df[pi_df['layer'] == _layer]['v']),
#     bins=32,
# )
# h2d

# fig = px.imshow(
#     img=h2d[0] / np.sum(h2d[0]),
#     x=(h2d[1][:-1] + h2d[1][1:]) / 2,
#     y=(h2d[2][:-1] + h2d[2][1:]) / 2,
#     aspect='auto',
#     template='plotly_dark',
# )
# fig

# # %% dist correl. per layer
# # layers = [0, 1, 2, 3, 9, 10]
# layers = list(range(11))
# h2ds = [
#     np.histogram2d(
#         pi_df[pi_df['layer'] == _layer]['dist'],
#         np.abs(pi_df[pi_df['layer'] == _layer]['v']),
#         bins=32,
#     )
#     # for _layer in range(11)
#     for _layer in layers
# ]
# h2ds

# imgs = np.array([d[0] for d in h2ds])
# fig = px.imshow(
#     imgs / np.max(imgs.sum(-1).sum(-1)),
#     # color_continuous_scale='gray',
#     facet_col=0,
#     facet_col_wrap=4,
#     labels={'facet_col' : 'layer'},
#     aspect='auto',
#     template='plotly_dark',
#     height=640,
#     width=1280,
# )
# # for i, v in enumerate(layers):
# #     fig.layout.annotations[i]['text'] = f'layer = {v}'
# # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.show()


# %% dist correl. 2
# layers = [0, 1, 2, 3, 9, 10]
# # layers = list(range(11))
# h2ds = [
#     np.histogram2d(
#         np.clip(_df['dist'], 0., 1.),
#         np.clip(np.abs(_df['v']), 0., 0.1),
#         bins=32,
#         range=[[0., 1.], [0., _max_v]]
#     )
#     for _layer in layers
#     for _head in range(4)
#     # for _df in [pi_df[pi_df['layer'] == _layer]]
#     for _df in [pi_df[(pi_df['layer'] == _layer) & (pi_df['head'] == _head)]]
#     for _max_v in [np.percentile(np.abs(_df['v']), 90) / 0.9]
# ]
# h2ds

# imgs = np.array([d[0] for d in h2ds])
# print(imgs.shape)
# fig = px.imshow(
#     img=imgs / np.max(imgs.sum(-1).sum(-1)),
#     x=(h2ds[0][1][:-1] + h2ds[0][1][1:]) / 2,
#     # y=(h2ds[0][2][:-1] + h2ds[0][2][1:]) / 2,
#     facet_col=0,
#     facet_col_wrap=4,
#     facet_col_spacing=0,
#     facet_row_spacing=0,
#     # labels={'facet_col' : ''},
#     aspect='auto',
# )
# fig.update_yaxes(showticklabels=False)
# fig.update_layout(
#     template='plotly_dark',
#     margin=dict(l=0,r=0,t=0,b=0),
#     # height=640,
#     height=800,
#     # width=800,
# )
# for i, v in enumerate(layers):
#     for j in range(4):
#         fig.layout.annotations[i * 4 + j]['text'] = f''
# fig.show()



# %%

# %% pi FLAT layers/heads
layers = [0, 1, 2, 3, 9, 10]
head_count = 4
_bin = 32
pi_abs = np.abs(pi)
_pi_abs = pi_abs[layers].reshape(-1, *pi_abs.shape[-2:])
# _pi_abs = np.clip(_pi_abs, 0., 0.1)
_pi_dig = np.digitize(
    _pi_abs,
    # np.linspace(np.min(_pi_abs), np.max(_pi_abs), _bin + 1),
    np.percentile(_pi_abs, np.linspace(0, 100, _bin + 1)[1:-1]),
)
print(f'{pi_abs.shape} -> {_pi_abs.shape}')


fig = px.imshow(
    # _pi_abs[:8],
    _pi_dig,
    facet_col=0,
    facet_col_wrap=head_count,
    facet_col_spacing=0,
    facet_row_spacing=0,
    # aspect='auto',
)

for i, v in enumerate(layers[::-1]):
    for j in range(head_count):
        fig.layout.annotations[i * head_count + j]['text'] = f'layer[{v}] head[{j}]'

fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    template='plotly_dark',
    margin=dict(l=0,r=0,t=20,b=0),
    # height=640,
    height=1200,
    width=1000,
)
fig.show()


# %%







# %% MAIN PLOTS
def add_grid(a, col, row=None, grid_value=None, sep=1):
    if row is None:
        row = col
    assert len(a.shape) == 2
    assert len(a.shape) >= 2
    assert a.shape[-2] % row == 0
    assert a.shape[-1] % col == 0
    cell_h = int(a.shape[-2] / row)
    cell_w = int(a.shape[-1] / col)
    a_4d = a.reshape(row, cell_h, col, cell_w)
    a_4d_t = a_4d.transpose(0,2,1,3)
    b = np.pad(
        a_4d_t,
        pad_width=[[0,0]] * 2 + [[0,sep]] * 2,
        constant_values=grid_value,
    )
    c = b.transpose(0,2,1,3).reshape(row*(cell_h+sep), col*(cell_w+sep))
    d = np.pad(
        c,
        pad_width=[[sep,0]] * 2,
        constant_values=grid_value,
    )
    return d

def img_concat(imgs, col=None, row=None, sep=0, border=0, bg=None):
    if isinstance(imgs, list):
        _count = len(imgs)
        assert len(imgs) > 0
        assert all([isinstance(v, np.ndarray) for v in imgs])
        assert all([len(v.shape) == len(list(imgs[0].shape)) for v in imgs[1:]])
        _shapes = np.array([v.shape for v in imgs]).astype(int)
        _shape = np.max(_shapes, axis=0)
        _pads = _shape[None] - _shapes
        _pads0 = np.floor(_pads / 2).astype(int)
        _pads1 = _pads - _pads0
        _pads_width = np.stack([_pads0, _pads1], axis=2)
        imgs_pad = [
            np.pad(
                _img,
                _pads_width[i],
                constant_values=bg,
            )
            for i, _img in enumerate(imgs)
        ]
        imgs = np.stack(imgs_pad, 0)
    assert isinstance(imgs, np.ndarray)
    assert len(imgs.shape) >= 3
    assert imgs.shape[0] >= 1
    _imgs = imgs
    if len(imgs.shape) > 3:
        _imgs = imgs.reshape(-1, *imgs.shape[-2:])
    _count = _imgs.shape[0]
    if col is None:
        if row is None:
            col = _count
            row = 1
        else:
            col = int(np.ceil(_count / row))
    else:
        if row is None:
            row = int(np.ceil(_count / col))
    
    if border is False:
        border = sep
    _shape = np.array(_imgs.shape[1:])
    assert len(_shape) == 2
    _img = np.zeros(
        shape=(_shape + sep) * np.array([row, col]) + sep * (border if border else -1),
        dtype=float,
    )
    _img[:, :] = bg
    for gy in range(row):
        for gx in range(col):
            gi = gy * col + gx
            if gi >= _count:
                break
            gp = np.array([gy, gx])
            pp = gp * (_shape + sep) + sep * (border if border else 0)
            # print(gp, pp, _shape)
            _img[
                pp[0] : pp[0] + _shape[0],
                pp[1] : pp[1] + _shape[1],
            ] = _imgs[gi]
    return _img

# %%
df = pd.DataFrame(
    columns=['v'],
    data=np.mean(np.random.sample([2000, 2000]) <= 0.5, axis=-1).reshape(-1, 1),
)
fig0 = px.histogram(
    df,
    x='v',
    template='plotly_dark',
)
fig0

# %%
# df_gm = px.data.gapminder().query("country in ['Canada', 'Botswana']")
# df_gm = px.data.gapminder().query("continent == 'Oceania'")
df_gm = px.data.gapminder().query("country in ['Canada', 'United States', 'Austria', 'Belgium']")
# df_gm = px.data.gapminder().query("continent == 'Americas'")

fig_gap_gdp_life = px.line(df_gm, x="lifeExp", y="gdpPercap", color="country", text="year")
fig_gap_gdp_life.update_traces(textposition="bottom right")
# fig_gap_gdp_life.show()

fig_gap_life = px.line(df_gm, x='year', y='lifeExp', color='country', symbol="country")
# fig_gap_life.show()

fig_gap_pop = px.line(df_gm, x='year', y='pop', color='country', symbol="country")
# fig_gap_pop.show()

for _fig in [fig_gap_gdp_life, fig_gap_life, fig_gap_pop]:
    _fig.update_layout(template='plotly_dark')
    _fig.show()

# %%
df_stock = px.data.stocks()
fig_stock = px.line(df_stock, x='date', y=["GOOG", 'AAPL', 'AMZN'])
fig_stock.update_layout(template='plotly_dark')
fig_stock.show()

# %%
IR = ImageReadme(image_width=240, begin_lines=[], dp='./plots/test_ir')

IR.save_figs(
    section='random',
    group='hist',
    sub_dir='hist',
    hist=fig0,
)
IR.save_figs(
    section='gapminder',
    group='correl',
    sub_dir='correl',
    gap_gdp_life=fig_gap_gdp_life,
)
IR.save_figs(
    section='gapminder',
    group='year',
    sub_dir='year',
    gap_life=fig_gap_life,
    gap_pop=fig_gap_pop,
)
IR.save_figs(
    section='stock',
    group='time',
    sub_dir='time',
    stock=fig_stock,
)
IR.save_readme()
IR


# %%
_dp_root = os.path.join('plots', 'pi_deit')
image_width = 240
readme_lines = [
    '## PLOTS',
]

# %% pi qk viz - centered
t = 14
t2 = 2 * t - 1
_bin = 100
_layer = 5
_head = 2

def plot_pi_qk_center(layer=0, head=0, with_q_grid=False):
    pi_qk = np.abs(pi)[layer, head, 1:, 1:].reshape(t, t, t, t)
    pi_qk_dig = np.digitize(
        pi_qk,
        np.percentile(
            pi_qk,
            np.linspace(0, 100, _bin + 1)[1:-1]
        ),
    ).astype(float)

    pi_qk_dig_center = pi_qk_dig

    pi_qk_dig_center_t = pi_qk_dig_center.transpose(0, 2, 1, 3)
    # pi_qk_dig_center_t_flat = pi_qk_dig_center_t.reshape(t*t2, t*t2)
    pi_qk_dig_center_t_flat = pi_qk_dig_center_t.reshape(t*t, t*t)

    pi_qk_dig_center_t_flat.shape

    _sep = 2
    _pi_qk_dig_center_t_flat = add_grid(
        pi_qk_dig_center_t_flat.astype(float),
        col=14,
        row=14,
        grid_value=None,
        sep=_sep,
    )
    fig = px.imshow(
        _pi_qk_dig_center_t_flat,
        color_continuous_scale='viridis',
    )
    fig.update_traces(
        # hovertemplate='x: %{x} <br> y: %{y} <br> z: %{z} <br> color: %{color}',
        hoverongaps=False,
    )
    axes_dict = dict(
        showticklabels=False,
        # tick0=-0.5,
        # dtick=(t2 + _sep) / 2,
        tick0=_sep,
        dtick=(t + _sep) + 1,
        tickwidth=0,
        # tickcolor='crimson',
        ticklen=0,
        gridwidth=4,
        gridcolor='white',
        
        showgrid=bool(with_q_grid),
        zeroline=False, # thick line at x=0
        visible=False,  # numbers below
    )
    fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
    fig.update_layout(
        # template='plotly_dark',
        margin=dict(l=0,r=0,t=20,b=0),
        # height=640,
        height=800,
        width=1000,
    )
    return fig

_name = f'deit_pi_qk_center'
_dp = os.path.join(_dp_root, _name)
if not os.path.isdir(_dp):
    os.makedirs(_dp)

for _layer in range(11):
    readme_lines.extend(['<p float="left" align="left">'])
    for _head in range(4):
        fig = plot_pi_qk_center(_layer, _head)
        _fn = f'{_name}_l{_layer}_h{_head}.png'
        _fp_rel = os.path.join(_name, _fn)
        fig.write_image(os.path.join(_dp, _fn))
        readme_lines.append(
            f'<img src="{_fp_rel}" width="{image_width}" />',
        )
    readme_lines.extend(['</p>'])

# fig.show()



# %% pi MASK qk - centered
t = 14
t2 = 2 * t - 1
_bin = 100
# _layer = 5
# _head = 2

def plot_pi_qk_mask(layer=0, head=0, prune=0.7, with_q_grid=False):
    pi_mask = np.digitize(
        pi,
        np.percentile(pi, [prune * 100]),
    )
    pi_mask.shape

    pi_qk_dig_center = pi_mask[layer, head, 1:, 1:].reshape(t, t, t, t).astype(float)

    # pi_qk_dig_center = pi_qk_dig

    pi_qk_dig_center_t = pi_qk_dig_center.transpose(0, 2, 1, 3)
    # pi_qk_dig_center_t_flat = pi_qk_dig_center_t.reshape(t*t2, t*t2)
    pi_qk_dig_center_t_flat = pi_qk_dig_center_t.reshape(t*t, t*t)

    pi_qk_dig_center_t_flat.shape

    _sep = 2
    _pi_qk_dig_center_t_flat = add_grid(
        pi_qk_dig_center_t_flat.astype(float),
        col=14,
        row=14,
        grid_value=None,
        sep=_sep,
    )
    fig = px.imshow(
        _pi_qk_dig_center_t_flat,
        color_continuous_scale='viridis',
    )
    fig.update_traces(
        # hovertemplate='x: %{x} <br> y: %{y} <br> z: %{z} <br> color: %{color}',
        hoverongaps=False,
    )
    axes_dict = dict(
        showticklabels=False,
        # tick0=-0.5,
        # dtick=(t2 + _sep) / 2,
        tick0=_sep,
        dtick=(t + _sep) + 1,
        tickwidth=0,
        # tickcolor='crimson',
        ticklen=0,
        gridwidth=4,
        gridcolor='white',
        # gridcolor='rgba(0,0,0,0)',
        
        showgrid=bool(with_q_grid),
        zeroline=False, # thick line at x=0
        visible=False,  # numbers below
    )
    fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
    fig.update_layout(
        # template='plotly_dark',
        margin=dict(l=0,r=0,t=20,b=0),
        # height=640,
        height=800,
        width=1000,
    )
    # fig = FF.format(
    #     fig
    # )
    return fig


# %%
prune = 0.7

_name = f'deit_pi_qk_mask_{int(prune * 100)}'
_dp = os.path.join(_dp_root, _name)
if not os.path.isdir(_dp):
    os.makedirs(_dp)

readme_lines.extend([
    f'> {_name}',
    # '<p float="left" align="left">',
])

for _layer in range(11):
    readme_lines.extend(['<p float="left" align="left">'])
    for _head in range(4):
        fig = plot_pi_qk_mask(_layer, _head)
        _fn = f'{_name}_l{_layer}_h{_head}.png'
        _fp_rel = os.path.join(_name, _fn)
        fig.write_image(os.path.join(_dp, _fn))
        readme_lines.append(
            f'<img src="{_fp_rel}" width="{image_width}" />',
        )
    readme_lines.extend(['</p>'])

# fig.show()



# %% pi qk viz - mean q
def get_pi_qk_mean_stack(
            pi_qk,
            dig=True,
            bin=100,
            plot=False,
            mask_min=1,
            t=14,
            ):
    _shape = pi_qk.shape
    assert len(_shape) == 4
    assert _shape[0] == _shape[1] == _shape[2] == _shape[3]
    
    pi_qk_mean = np.zeros([t * 2 - 1, t * 2 - 1], dtype=np.float32)
    pi_qk_mean_mask = np.zeros([t * 2 - 1, t * 2 - 1], dtype=np.int32)
    for qy in range(t):
        for qx in range(t):
            cy = t - 1 - qy
            cx = t - 1 - qx
            pi_qk_mean[
                cy : cy + t,
                cx : cx + t,
            ] += pi_qk[qy, qx]
            pi_qk_mean_mask[
                cy : cy + t,
                cx : cx + t,
            ] += 1
    
    pi_qk_mean = pi_qk_mean / np.clip(pi_qk_mean_mask, mask_min, t*t)
    r = pi_qk_mean
    if dig:
        pi_qk_mean_dig = np.digitize(
            pi_qk_mean,
            np.percentile(pi_qk_mean, np.linspace(0, 100, bin+1)[1:-1]),
        ) / bin
        r = pi_qk_mean_dig
    if plot:
        # px.imshow(pi_qk_mean, template='plotly_dark')
        # px.imshow(pi_qk_mean_dig, template='plotly_dark')
        fig = px.imshow(r, template='plotly_dark')
        fig.show()
    return r


# %% pi qk MEAN (all head, all layer)
def plot_pi_qk_mean(pi, layers=11, heads=4, t=14, dig=False, bin=100):
    pi_qk_mean = []
    if isinstance(layers, int):
        name = f'0-{layers}'
        layers = list(range(layers))
    else:
        assert isinstance(layers, list)
        name = '_'.join([str(v) for v in layers])
    for li, l in enumerate(layers):
        for h in range(heads):
            _pi_qk = pi[l, h, 1:, 1:].reshape(t, t, t, t)
            _pi_qk_mean = get_pi_qk_mean_stack(
                _pi_qk,
                dig=dig,
                bin=bin,
                t=t,
            )
            pi_qk_mean.append(_pi_qk_mean)
    
    pi_qk_mean = np.stack(pi_qk_mean)
    pi_qk_mean = np.clip(pi_qk_mean / np.percentile(pi_qk_mean, 99), 0, 1)
    pi_qk_mean_img = img_concat(
        pi_qk_mean,
        col=heads,
        sep=4,
    )
    pi_qk_mean_img.shape

    fig = px.imshow(
        pi_qk_mean_img,
        zmin=0,
        zmax=1,
        color_continuous_scale='viridis',
    )
    fig.update_traces(
        # hovertemplate='x: %{x} <br> y: %{y} <br> z: %{z} <br> color: %{color}',
        hoverongaps=False,
    )
    axes_dict = dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        # gridcolor='white',
    )
    fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
    _scale = 6
    _layer_count = len(layers)
    fig.update_layout(
        # template='plotly_dark',
        margin=dict(l=0,r=0,t=0,b=0),
        # height=640,
        height=(_layer_count * 31 - 4) * _scale,
        width=(heads * 31 - 4) * _scale + 100,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    # fig = FF.format(
    #     fig
    # )
    return fig

# %%
_name = f'deit_pi_qk_mean'
fig = plot_pi_qk_mean(pi, layers=11, heads=4, t=14, dig=False, bin=100)
fig.show()
fig.write_image(f'plots/pi_deit/{_name}.png')
readme_lines.extend([
    f'> {_name}',
    '<p float="left" align="left">',
    f'<img src="{f"{_name}.png"}" width="{image_width}" />',
    '</p>',
])

# %%
fig = plot_pi_qk_mean(pi, layers=11, heads=4, t=14, dig=True, bin=100)
fig.show()


# %%
_layers = [0, 2, 4, 6, 8, 10]
_name = f'deit_pi_qk_mean_{"_".join([str(v) for v in _layers])}'
fig = plot_pi_qk_mean(pi, _layers, 4, 14, False)
fig.show()
fig.write_image(f'plots/pi_deit/{_name}.png')
readme_lines.extend([
    f'> {_name}',
    '<p float="left" align="left">',
    f'<img src="{f"{_name}.png"}" width="{image_width}" />',
    '</p>',
])

# %% save README.md
fp_rm = os.path.join(_dp_root, f'README.md')

readme_txt = '\n\n'.join(readme_lines)

with open(fp_rm, 'w') as fo:
    fo.writelines(readme_txt)
print(f'[PLOT] saved with README.md in <{_dp_root}>')


# %%
def scale_map(a, range=(0, 1), axis=None):
    _min = np.min(a, axis=axis, keepdims=True)
    _max = np.max(a, axis=axis, keepdims=True)
    b = (a - _min) / np.clip(_max - _min, 0.000000001, None)
    return b * (range[1] - range[0]) + range[0]

# %% pi all cls
H = 4
pi_cls_q = pi[:, :, 0, 1:].reshape(-1, 14, 14)
pi_cls_k = pi[:, :, 1:, 0].reshape(-1, 14, 14)

# pi_cls_qk = pi[:, :, 0, 0]
# pi_cls_q.shape, pi_cls_k.shape, pi_cls_qk.shape

# pi_cls_q = pi_cls_q / np.max(pi_cls_q, axis=0, keepdims=True)
# pi_cls_k = pi_cls_k / np.max(pi_cls_k, axis=0, keepdims=True)

pi_cls_q = scale_map(pi_cls_q, range=(0, 1), axis=(1, 2))
pi_cls_k = scale_map(pi_cls_k, range=(0, 1), axis=(1, 2))

pi_cls_q.shape, pi_cls_k.shape


pi_cls_q_img = img_concat(
    pi_cls_q,
    col=H,
    sep=1,
)
pi_cls_k_img = img_concat(
    pi_cls_k,
    col=H,
    sep=1,
)
pi_cls_img = img_concat(
    [pi_cls_q_img, pi_cls_k_img],
    # [pi_cls_q_img, pi_cls_k_img, pi_cls_qk],
    sep=10,
)
# fig = px.imshow(pi_cls_q_img, height=800)
fig = px.imshow(pi_cls_img)

fig.update_traces(
    hoverongaps=False,
)
axes_dict = dict(
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    showline=False,
    # gridcolor='white',
)
fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
fig.update_layout(
    template='plotly_dark',
    margin=dict(l=0,r=0,t=0,b=0),
    # height=640,
    height=800,
    # width=1000,
)

_name = f'deit_pi_cls_q+k'
fig.write_image(f'plots/pi_deit/{_name}.png')
fig

# %% pi last cls
# a = np.abs(pi)[-1, :, 0, 1:].reshape(-1, 14, 14)
# # print(a.shape)
# b = img_concat(
#     a,
#     col=2,
#     sep=3,
# )
# px.imshow(b, height=800, template='plotly_dark')







# %% temp mean qk across q
# a = pi[0, 2, 1:, 1:]
# a = a.reshape(-1, 14, 14)
# b = np.mean(a, axis=0)
# px.imshow(b)










# %%
