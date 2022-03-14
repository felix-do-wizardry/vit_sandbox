# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
def plot_pi(num_heads, num_layers, data, lm = False, prune = 0.0, type_ = None):
    fig, ax = plt.subplots(num_layers,num_heads, figsize = (20,40))
    for j in range(num_layers):
        for i in range(num_heads):
            if lm:
                if prune == 0.:
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
def plot_pi_swin(data, num_heads=138, prune=0.0, cols=3):
    rows = int(np.ceil(num_heads / cols))
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(20,int(np.ceil(20 / cols * rows))),
    )
    for y in range(rows):
        for x in range(cols):
            hi = y * cols + x
            a = np.abs(data[hi,:,:])
            
            if prune > 0.:
                # if pruning, only draw the mask
                a = a != 0.
            
            ax[y, x].imshow(a/max(a.max(), 0.000000001),cmap='viridis')
    
    return fig, ax

# %%
fps_np = [
    'pi/pi_coco.npy',
]
fp_np = fps_np[0]
pi_raw = np.load(fp_np)[0]
swin_heads = [3] * 2 + [6] * 2 + [12] * 6 + [24] * 2
pi_raw.shape

# %%
fig, ax = plot_pi_swin(
    data=pi_raw,
    num_heads=138,
    prune=0.7,
    cols=6,
)


# %%
# pi_deit_raw = np.load('pi/mixed_prune_pi_mask_60_15.npy')[0]
pi_deit_raw = np.load('pi/attention_score_pi_mask_60.npy')[0]
pi_deit_raw.shape


fig, ax = plot_pi_swin(
    data=pi_deit_raw[:24],
    num_heads=24,
    prune=0.7,
    cols=6,
)

# %%










# %%
import os
import numpy as np
import plotly.express as px

# %%
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

# %%
# a = pi_raw[14]
# px.imshow(
#     a,
#     zmin=a.min(),
#     zmax=a.max(),
#     template='plotly_dark',
# )

# %%
dp = 'attn'
names = [
    'baseline',
    'gmm_gd_0.65',
    'gmm_gd_0.7_0.15',
]
cls_attn = np.stack([
    np.load(os.path.join(dp, f'{name}_attn_cls.npy'))
    for name in names
])
cls_attn_exp = np.exp(cls_attn)
# cls_attn = cls_attn_exp / np.sum(cls_attn_exp, axis=-1, keepdims=True)
t = int(np.sqrt(cls_attn.shape[-1] - 1))
# assert t * t + 1 == cls_attn.shape[-1]
Z, M, H, Nk = cls_attn.shape
cls_attn_img = cls_attn[..., 1:].reshape([Z, M, H, t, t])

# [3, M, H, Nk], [3, M, H, t, t]
cls_attn.shape, cls_attn_img.shape, t



# %%
a = cls_attn_img[0, 0, 1]
px.imshow(
    a,
    zmin=a.min(),
    zmax=a.max(),
    template='plotly_dark',
)

# %%

# %%
img_count = 4
img_offset = 25
imgs = cls_attn_img[0, img_offset : img_offset + img_count, :]
imgs = imgs / imgs.max(-1).max(-1)[..., None, None]

px.imshow(
    img_concat(
        imgs=imgs.reshape(-1, t, t),
        col=4,
        sep=1,
    ),
    template='plotly_dark',
    height=800,
)
# %%
img_count = 4
img_offset = 25
imgs = cls_attn_img[0, img_offset : img_offset + img_count, :].reshape(-1, t, t)
imgs = imgs / imgs.max(-1).max(-1)[..., None, None]
imgs = imgs.repeat(4, -1).repeat(4, -2)

px.imshow(
    img_concat(
        imgs=imgs,
        col=4,
        sep=1,
    ),
    template='plotly_dark',
    height=800,
)








# %% load imagenet data
import torch
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# %%
input_size = 224
input_size = 14 * 4
trans = transforms.Compose([
    transforms.Resize(int((256 / 224) * input_size), interpolation=3),  # to maintain same ratio w.r.t. 224 images
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    # transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])
ds = datasets.ImageFolder('/host/ubuntu/data/imagenet2012/val', transform=trans)
ds

# %%
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=1,
    shuffle=False,
    # sampler=None,
    # batch_sampler=None, num_workers=4, collate_fn=None,
    # pin_memory=False, drop_last=False, timeout=0,
    # worker_init_fn=None, *, prefetch_factor=2,
    # persistent_workers=False,
)
len(dl)

# %%
# features, labels = next(iter(dl))

# %%
inputs = []
labels = []
for i, d in enumerate(dl):
    if i >= 200:
        break
    # print(d)
    inputs.append(d[0].detach().cpu().numpy())
    labels.append(d[1].detach().cpu().numpy())

inputs = np.concatenate(inputs, axis=0)
labels = np.concatenate(labels, axis=0)
inputs = inputs.transpose(0, 2, 3, 1)
inputs.shape, labels.shape

# %%
px.imshow(
    (inputs[0] * 255).astype(np.uint8),
)

# %%
px.imshow(
    np.mean(inputs[0], axis=-1),
    binary_string=True,
    template='plotly_dark',
)





# %%
img_count = 4
img_offset = 25
img_count = 4
img_offset = 100
indices = [20, 60, 110, 160]
indices = [30, 80, 130, 180]
indices = [130]

# _inputs = np.mean(inputs[img_offset : img_offset + img_count], axis=-1)
_inputs = np.mean(inputs[indices], axis=-1)
_inputs.shape

# imgs_cls = cls_attn_img[0, img_offset : img_offset + img_count, :]
imgs_cls = cls_attn_img[:1, indices, 0]
_max = imgs_cls.max(-1).max(-1)[..., None, None]
_min = imgs_cls.min(-1).min(-1)[..., None, None]
imgs_cls = (imgs_cls - _min) / (_max - _min)
imgs_cls = imgs_cls.repeat(input_size//14, -1).repeat(input_size//14, -2)
imgs_cls.shape


imgs = np.concatenate([_inputs[:, None], imgs_cls], axis=1)
imgs = imgs.reshape(-1, input_size, input_size)
imgs.shape

_img = img_concat(imgs, col=5, sep=2)
_img.shape


fig = px.imshow(
    _img,
    # binary_string=True,
    color_continuous_scale='gray',
    template='plotly_dark',
)
fig.update_layout(
    margin=dict(t=0,b=0,l=0,r=0),
    showlegend=False,
)

# %%








# %%

# %%
df_val = pd.read_csv('/host/ubuntu/data/imagenet2012/val_labels.csv')
df_val

df_map = pd.read_csv('/host/ubuntu/data/imagenet2012/class_map.csv')
df_map

# %%
import shutil
dp_src = '/host/ubuntu/data/imagenet2012/val_flat'
dp_dest = '/host/ubuntu/data/imagenet2012/val'

class_codes = list(df_map['code'])

for i, _code in enumerate(class_codes):
    _dp = os.path.join(dp_dest, _code)
    if not os.path.isdir(_dp):
        os.makedirs(_dp)

for i, d in enumerate(df_val.to_dict('records')):
    _code = class_codes[d['class']]
    fp_src = os.path.join(dp_src, d['image'])
    fp_dest = os.path.join(dp_dest, _code, d['image'])
    assert os.path.isfile(fp_src)
    shutil.copyfile(fp_src, fp_dest)
    if (i + 1) % 100 == 0:
        print(f'\r[{i+1}/50k]', end='')



# %%