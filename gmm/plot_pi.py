# %%
import pickle
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def read_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

# %%
# pi_lm = read_pkl('pi/pi_lm.pkl')
# pi_lm.keys()

# %%
pi_pkl = read_pkl('pi/pi_img_deit.pkl')
pi_pkl.keys()

# %%
# [a[0.0][k].shape for k in sorted(a[0.0].keys())]
pi = np.array([pi_pkl[0.0][k].squeeze(0) for k in sorted(pi_pkl[0.0].keys())])
pi.shape

# %%
pi5 = np.array([pi_pkl[0.5][k].squeeze(0) for k in sorted(pi_pkl[0.5].keys())])
pi5.shape

# %%
np.mean((pi5 * (pi == pi5)) == 0)

# %%
np.percentile(pi, np.linspace(0, 100, 11))

# %%
pi_hists = [
    np.histogram(
        bins=100,
        # a=v.reshape(-1),
        # range=(b.min(), b.max()),
        # a=v.reshape(-1),
        # range=(-0.1, 0.1),
        a=np.abs(v.reshape(-1)),
        range=(0, 0.1),
    )
    for v in pi[:-1]
]
pi_hist_df = pd.DataFrame([
    {
        'layer': i,
        'value': _value,
        'count': _count,
    }
    for i, (count, values) in enumerate(pi_hists)
    for _count, _value in zip(count, (values[:-1] + values[1:]) / 2)
])
pi_hist_df

# %%
fig = px.line(
    pi_hist_df,
    x='value',
    y='count',
    color='layer',
)
fig.update_layout(
    template='plotly_dark',
    yaxis=dict(type='log'),
)
fig


# %%
def index2pos(indices, t=14, cls_count=1):
    # p = np.zeros(len(indices), dtype=float)
    p = np.stack([
        np.floor((indices - cls_count) / t),
        (indices - cls_count) % t,
    ], axis=1)
    indices_cls = np.where(indices < cls_count)[0]
    p[indices_cls] = [[t/2, t/2] for _ in range(len(indices_cls))]
    return p

def token_dist(p0, p1):
    return np.sqrt(np.sum((p0 - p1) ** 2, axis=-1))

# %%
pi_df = pd.DataFrame([
    {
        'v': np.abs(v3),
        'q': _q,
        'k': _k,
        # 'dist': token_dist(
        #     index2pos(_q, 14, 1),
        #     index2pos(_k, 14, 1),
        # ),
        'layer': _layer,
        'head': _head,
    }
    for _layer, v0 in enumerate(pi)
    for _head, v1 in enumerate(v0)
    for _q, v2 in enumerate(v1)
    for _k, v3 in enumerate(v2)
])
pi_df['dist'] = token_dist(
    index2pos(np.array(pi_df['q']), 14, 1),
    index2pos(np.array(pi_df['k']), 14, 1),
) / (13 * np.sqrt(2))
pi_df

# %%


# %% dist correlation np

# %%
_layer = 9
h2d = np.histogram2d(
    pi_df[pi_df['layer'] == _layer]['dist'],
    np.abs(pi_df[pi_df['layer'] == _layer]['v']),
    bins=32,
)
h2d

fig = px.imshow(
    img=h2d[0] / np.sum(h2d[0]),
    x=(h2d[1][:-1] + h2d[1][1:]) / 2,
    y=(h2d[2][:-1] + h2d[2][1:]) / 2,
    aspect='auto',
    template='plotly_dark',
)
fig

# %% dist correl. per layer
# layers = [0, 1, 2, 3, 9, 10]
layers = list(range(11))
h2ds = [
    np.histogram2d(
        pi_df[pi_df['layer'] == _layer]['dist'],
        np.abs(pi_df[pi_df['layer'] == _layer]['v']),
        bins=32,
    )
    # for _layer in range(11)
    for _layer in layers
]
h2ds

imgs = np.array([d[0] for d in h2ds])
fig = px.imshow(
    imgs / np.max(imgs.sum(-1).sum(-1)),
    # color_continuous_scale='gray',
    facet_col=0,
    facet_col_wrap=4,
    labels={'facet_col' : 'layer'},
    aspect='auto',
    template='plotly_dark',
    height=640,
    width=1280,
)
# for i, v in enumerate(layers):
#     fig.layout.annotations[i]['text'] = f'layer = {v}'
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.show()


# %% dist correl. 2
layers = [0, 1, 2, 3, 9, 10]
# layers = list(range(11))
h2ds = [
    np.histogram2d(
        np.clip(_df['dist'], 0., 1.),
        np.clip(np.abs(_df['v']), 0., 0.1),
        bins=32,
        range=[[0., 1.], [0., _max_v]]
    )
    for _layer in layers
    for _head in range(4)
    # for _df in [pi_df[pi_df['layer'] == _layer]]
    for _df in [pi_df[(pi_df['layer'] == _layer) & (pi_df['head'] == _head)]]
    for _max_v in [np.percentile(np.abs(_df['v']), 90) / 0.9]
]
h2ds

imgs = np.array([d[0] for d in h2ds])
print(imgs.shape)
fig = px.imshow(
    img=imgs / np.max(imgs.sum(-1).sum(-1)),
    x=(h2ds[0][1][:-1] + h2ds[0][1][1:]) / 2,
    # y=(h2ds[0][2][:-1] + h2ds[0][2][1:]) / 2,
    facet_col=0,
    facet_col_wrap=4,
    facet_col_spacing=0,
    facet_row_spacing=0,
    # labels={'facet_col' : ''},
    aspect='auto',
)
fig.update_yaxes(showticklabels=False)
fig.update_layout(
    template='plotly_dark',
    margin=dict(l=0,r=0,t=0,b=0),
    # height=640,
    height=800,
    # width=800,
)
for i, v in enumerate(layers):
    for j in range(4):
        fig.layout.annotations[i * 4 + j]['text'] = f''
fig.show()



# %%

# %% pi viz layers/heads
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


# %%
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
    assert len(imgs.shape) >= 3
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

# a = np.random.randint(10, size=[6, 2, 2]).astype(float)
# img_concat(a, col=3, sep=1)

# %% pi qk viz (plain)
# t = 14
# _bin = 100
# _layer = 2
# _head = 2
# pi_qk = np.abs(pi)[_layer, _head, 1:, 1:].reshape(t, t, t, t)

# pi_qk_dig = np.digitize(
#     pi_qk,
#     np.percentile(
#         pi_qk,
#         np.linspace(0, 100, _bin + 1)[1:-1]
#     ),
# ).astype(float)
# pi_qk_dig_t = pi_qk_dig.transpose(0, 2, 1, 3)
# pi_qk_dig_t_flat = pi_qk_dig_t.reshape(t*t, t*t)
# pi_qk.shape, pi_qk_dig_t.shape, pi_qk_dig_t_flat.shape

# # %%
# _pi_qk_img_flat_dig = add_grid(
#     pi_qk_dig_t_flat,
#     col=14,
#     row=14,
#     grid_value=None,
#     sep=1,
# )
# fig = px.imshow(
#     _pi_qk_img_flat_dig,
# )

# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)
# fig.update_layout(
#     template='plotly_dark',
#     margin=dict(l=0,r=0,t=20,b=0),
#     # height=640,
#     height=800,
#     width=1000,
# )
# fig.show()

# %%




# %% pi qk viz - centered
t = 14
t2 = 2 * t - 1
_bin = 100
_layer = 4
_head = 1
pi_qk = np.abs(pi)[_layer, _head, 1:, 1:].reshape(t, t, t, t)
pi_qk_dig = np.digitize(
    pi_qk,
    np.percentile(
        pi_qk,
        np.linspace(0, 100, _bin + 1)[1:-1]
    ),
).astype(float)

# pi_qk_dig_center = np.array([
#     [
#         np.pad(
#             pi_qk_dig[qy, qx],
#             pad_width=[[t - 1 - qy, qy], [t - 1 - qx, qx]],
#             constant_values=None,
#         )
#         for qx in range(pi_qk_dig.shape[1])
#     ]
#     for qy in range(pi_qk_dig.shape[0])
# ])
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
)
fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
fig.update_layout(
    template='plotly_dark',
    margin=dict(l=0,r=0,t=20,b=0),
    # height=640,
    height=800,
    width=1000,
)
fig.show()

# %%





# %% pi qk viz - mean q
def get_pi_qk_mean_stack(
            pi_qk,
            dig=True,
            bin=100,
            plot=False,
            mask_min=1,
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
        )
        r = pi_qk_mean_dig
    if plot:
        # px.imshow(pi_qk_mean, template='plotly_dark')
        # px.imshow(pi_qk_mean_dig, template='plotly_dark')
        fig = px.imshow(r, template='plotly_dark')
        fig.show()
    return r

# %%
t = 14
t2 = 2 * t - 1
_bin = 100
_layer = 2
_head = 1
pi_qk = np.abs(pi)[_layer, _head, 1:, 1:].reshape(t, t, t, t)

H = 4
pi_qk_mean_digs = []
for l in range(11):
    for h in range(H):
        _pi_qk = np.abs(pi)[l, h, 1:, 1:].reshape(t, t, t, t)
        pi_qk_mean_dig = get_pi_qk_mean_stack(
            _pi_qk,
            dig=True,
        )
        pi_qk_mean_digs.append(pi_qk_mean_dig)
        # fig = px.imshow(pi_qk_mean_dig, template='plotly_dark')


pi_qk_mean_digs = np.stack(pi_qk_mean_digs)
pi_qk_mean_digs_img = img_concat(
    pi_qk_mean_digs,
    col=H,
    sep=4,
)
pi_qk_mean_digs_img.shape

fig = px.imshow(
    pi_qk_mean_digs_img,
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
    # tick0=_sep,
    # dtick=(t + _sep) + 1,
    # tickwidth=0,
    # # tickcolor='crimson',
    # ticklen=0,
    # gridwidth=4,
    # gridcolor='white',
)
fig.update_xaxes(**axes_dict).update_yaxes(**axes_dict)
_scale = 4
fig.update_layout(
    # template='plotly_dark',
    margin=dict(l=0,r=0,t=0,b=0),
    # height=640,
    height=337 * _scale,
    width=120 * _scale + 100,
    plot_bgcolor='rgba(0,0,0,0)',
)
fig.show()
fig.write_image('plots/deit_pi_qk_mean.png')


# %% pi last cls
# pi_qk = np.abs(pi)[-1, _head, 1:, 1:].reshape(t, t, t, t)
# pi_qk.shape
# pi_qk[]
a = np.abs(pi)[-1, :, 0, 1:].reshape(-1, 14, 14)
# print(a.shape)
b = img_concat(
    a,
    col=2,
    sep=3,
)

px.imshow(b, height=800, template='plotly_dark')


# %%








# %%
# plot_pi(8,16,pi_lm, lm = True)

# %%

import matplotlib.pyplot as plt

def plot_pi(num_heads, num_layers, data, lm = False, prune = 0.0, type_ = None):
    fig, ax = plt.subplots(num_layers,num_heads, figsize = (20,40))
    for j in range(num_layers):
        for i in range(num_heads):
            if lm:
                if prune ==0.:
                    a = np.tril(data[prune][j][:,:,0,i])
                else:
                    a = np.tril(data[f'{prune}_{type_}'][j][:,:,0,i]) != 0.
            else:
                if prune == 0.:
                    a = data[prune][j][0,i,:,:] 
                else:
                    # if pruning, only draw the mask
                    a = data[prune][j][0,i,:,:] != 0.
            ax[j, i].imshow(a/a.max(),cmap='viridis')
# plot_pi(8,16,pi_lm, lm = True)

# %%


# %%
# pi_img = read_pkl('pi/pi_img_deit.pkl')
# plot_pi(4,12,pi_img, prune = 0.0)

# %%
# pi_img = read_pkl('pi/pi_img_deit.pkl')
# plot_pi(4,12,pi_img, prune = 0.7)

# %%
###### pruning LM

from torch import nn
import torch

class Layer(nn.Module):
    def __init__(self, pi):
        super().__init__()
        self.pi = nn.Parameter(torch.tensor(pi), requires_grad= True)

class Model(nn.Module):
    def __init__(self, data, normalized = False):
        super().__init__()
        self.layers = []
        # for i in range(16):
            # layer = Layer()
            # layer.pi = torch.tensor(data[0.0][i])
        for i in range(16):
            if not normalized:
                self.layers.append(Layer(data[0.0][i]))
            else:
                pi = abs(data[0.0][i])
                pi_tril = pi*(torch.tril((torch.ones_like(torch.tensor(pi[:,:,0,0])))).view(256,256,1,1).expand(-1,-1,-1,8).numpy())
                pi = pi/(pi_tril.sum(1, keepdims = True)/np.expand_dims(np.arange(1,257), 1) + 1e-6)
                self.layers.append(Layer(pi))


# %%
model = Model(pi_lm)

# %%
from torch.nn.utils import prune
def pruning(amount, global_prune=False):
    amount = amount*0.5
    ####################### GLOBAL PRUNING ###############################
    if global_prune:
        print(f'GLOBAL PRUNING THE MODEL WITH AMOUNT {amount*2}')
        parameters_to_prune = []
 
        for i in range(16):
            #Set upper half to inf
            model.layers[i].pi.data += torch.triu((model.layers[i].pi.data[:,:,0,0])*float(100000), diagonal=1).view(256,256,1,1).expand(-1,-1,-1,8)
            parameters_to_prune.append((model.layers[i], 'pi'))
 
        #Prune
        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount = amount)
       
        for i in range(16):
            ### remove params
            pi_mask = model.layers[i].pi_mask.data
            prune.remove(model.layers[i], 'pi')
            model.layers[i].register_buffer('pi_mask', pi_mask)
 
    ############### LOCAL PRUNING #######################
    else:
        # print("//////",torch.cuda.device_count())
        print(f'LOCAL PRUNING THE MODEL WITH AMOUNT {amount*2}')
 
        for i in range(16):
            ### pruning locally layer by layer
 
            # Set upper half to inf
            model.layers[i].pi.data += torch.triu((model.layers[i].pi.data[:,:,0,0])*float(100000), diagonal=1).view(256,256,1,1).expand(-1,-1,-1,8)
           
            # prune step
            prune.l1_unstructured(model.layers[i], name = 'pi', amount = amount)
           
            #set upper half to 0
            model.layers[i].pi.data *= torch.tril(torch.ones(256,256)).view(256,256,1,1).expand(-1,-1,-1,8)
           
            ### remove params
            pi_mask = model.layers[i].pi_mask.data
            # print(pi_mask[:,:,0,0])
            # assert 1==2
            prune.remove(model.layers[i], 'pi')
            model.layers[i].register_buffer('pi_mask', pi_mask)

# %%
from collections import defaultdict
def save_pi(file,amount, type_ = 'local'):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except:
        EOFError
        data = defaultdict()
    # else:
    #     if        
    pis = defaultdict()
    for i in range(16):
        pi = model.layers[i].pi.data.cpu().numpy()
        pis[i] = pi
    if data is not None:
        data.update({f'{amount}_{type_}': pis})
    else:
        data = pis
    with open(file, 'wb') as f:
        pickle.dump(data, f)

# %%
# type_ = 'global'
# for amount in [0.4, 0.5, 0.6,0.7]:
#     model = Model(pi_lm)
#     is_global = type_ == 'global'
#     pruning(amount, is_global)
#     save_pi('pi/pi_lm.pkl',amount,type_)

# %%
# pi_lm.keys()
# pi_lm[0.0] = pi_lm[0.6]
# del pi_lm[0.6]
# with open('pi/pi_lm.pkl', 'wb') as f:
#     pickle.dump(pi_lm, f)

# %%
# pi_lm = read_pkl('pi/pi_lm.pkl')

# %%


# %%


# %%
plot_pi(8,16,pi_lm, lm = True, prune=0.7, type_= 'local')

# %%
plot_pi(8,16,pi_lm, lm = True, prune=0.4, type_= 'global')

# %%



