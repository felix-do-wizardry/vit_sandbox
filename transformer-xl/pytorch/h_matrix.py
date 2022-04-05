# %%
import numpy as np
import json, os, time

# import torch
# import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go


# %%
MASK_TYPES = [
    'h1d',
    # 'h', 'hdist',
    # 'dist', 'distq',
    # 'cross', 'crossq',
    # 'col', 'row',
    # 'x',
]
MASK_TYPES_DIST = [
    'dist', 'distq',
    # 'cross', 'crossq',
]

# CLS_TOKEN_TYPES = [
#     'copy',
#     'pos',
#     'mask',
#     'sum',
# ]

# %% DEV
# n = 150
# level = 2

# m = np.zeros([n, n], dtype=int)
# for i in range(level):
#     j = i + 1
#     _s = int(n // 2 ** j)
#     # print(i, j, _s, list(range(0, n, _s)))
#     for r0 in range(0, n, _s):
#         r1 = r0 + _s
#         m[r0 : r1, r0 : r1] = j

# m.shape

# fig = px.imshow(
#     m,
#     template='plotly_dark',
# )
# fig.update_layout(
#     margin=dict(t=0,b=0,l=0,r=0),
#     height=600,
#     width=600,
# )
# fig.show()

# m.shape, n, level, n % 2**level

# %%
class H_Matrix_1D:
    '''1D H_Matrix for Transformer Attention
    '''
    def __init__(self,
                n=150,
                level=2,
                mask_type='h1d',
                # **kwargs,
                ):
        '''Args:
        n (int): number of tokens (sequence length)
        level (int): level of h matrix (== mask_levels - 1)
        '''
        
        assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        self.mask_type = mask_type
        self.is_dist = mask_type in MASK_TYPES_DIST
        
        assert isinstance(n, int)
        assert n >= 2
        assert isinstance(level, int)
        assert level > 0
        assert n % (2**level) == 0
        
        self.n = n
        self.level = level
        
        n = 160
        level = 2
        # if not self.is_dist:
        #     assert level % 2 == 0
        
        m = np.zeros([n, n], dtype=int)
        for i in range(level):
            j = i + 1
            _s = int(n // 2 ** j)
            for r0 in range(0, n, _s):
                r1 = r0 + _s
                m[r0 : r1, r0 : r1] = j
        
        # shape = [t*t+1, t*t+1], range = [-1, level]
        self.m = m
        # self.dist = dist
        # self.dist_dig = dist_dig
        # self.distq_dig = distq_dig
        
        
        if mask_type in ['h1d']:
            self.indexed_mask = self.m
        # elif mask_type in ['dist']:
        #     self.indexed_mask = self.dist_dig
        # elif mask_type in ['distq']:
        #     self.indexed_mask = self.distq_dig
        else:
            raise NotImplementedError(f'`mask_type`[{mask_type}] has not been implemented')
    
    @classmethod
    def get_dist(cls, p0, p1, t):
        return np.sqrt(np.sum((p0 - p1) ** 2)) / t


# %%
class H_Matrix_Masks_1D:
    '''H_Matrix_Masks_1D
    '''
    def __init__(self,
                mask_type='h1d',
                n=160,
                mask_levels=3,
                ):
        
        self._mask_type = mask_type
        self._n = n
        self._mask_levels = mask_levels
        self.masks, self.mask_levels_final = self.get_masks(
            mask_type=mask_type,
            n=n,
            mask_levels=mask_levels,
        )
    
    
    @classmethod
    def get_masks(cls,
                mask_type='h1d',
                n=160,
                mask_levels=3,
                ):
        
        assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        token_count = n
        
        hm = H_Matrix_1D(
            n=n,
            level=mask_levels - 1,
            mask_type=mask_type,
        )
        
        indexed_mask = hm.indexed_mask
        mask_levels_final = mask_levels
        
        # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
        masks = np.array([
                indexed_mask == i
                for i in range(mask_levels)
            ],
            dtype=np.float32,
        ).transpose(1, 2, 0)[None, None]
        
        _shape = list(indexed_mask.shape)
        assert (len(_shape) == 2 and _shape[0] == _shape[1] == token_count
            ), f'indexed_mask.shape={_shape} != token_count[{token_count}]'
        
        print(f'type[{mask_type}] global[{"?"}] mask_levels_final[{mask_levels_final}]')
        print(f'masks{list(masks.shape)}')
        
        return masks, mask_levels_final
    
    @classmethod
    def plot_masks(cls,
                masks,
                # mask_type='distq',
                # cls_token_type='pos',
                # cls_token_pos=0.5,
                cls_token_count=1,
                token_grid_size=14,
                # mask_levels=3,
                mask_levels_final=3,
                grid_sep=2,
                plot_size=[800, 800],
                ):
        import plotly.express as px
        # import plotly.graph_objects as go
        
        t = token_grid_size
        token_count = int(token_grid_size ** 2) + cls_token_count
        
        a = masks.reshape(token_count, token_count, mask_levels_final)
        a = np.sum(a * np.arange(mask_levels_final).reshape(1, 1, -1), -1)
        b = a[cls_token_count:, cls_token_count:].reshape(t, t, t, t)
        _sep = int(grid_sep)
        _img = np.full([t * (t + _sep) - _sep] * 2, fill_value=None)
        
        for qy in range(t):
            for qx in range(t):
                _y = qy * (t + _sep)
                _x = qx * (t + _sep)
                _img[
                    _y : _y + t,
                    _x : _x + t,
                ] = b[qy, qx]
        
        # b = a.transpose(0, 2, 1, 3)
        # b = b.reshape(t*t, t*t)
        a.shape, b.shape, _img.shape
        
        fig_grid = px.imshow(
            # b,
            _img,
            labels=dict(
                x="Q column",
                y="Q row",
                # color="Mask Index",
            ),
            y=(np.arange(_img.shape[0]) - t/2 - 0.5) / (t + _sep),
            x=(np.arange(_img.shape[1]) - t/2 - 0.5) / (t + _sep),
        )
        fig_grid.update_layout(
            template='plotly_dark',
            margin=dict(t=0,b=0,l=0,r=0),
            height=plot_size[0],
            width=plot_size[1],
        )
        # fig_grid.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_grid.update_xaxes(tick0=0, dtick=1).update_yaxes(tick0=0, dtick=1)
        
        fig_mask = px.imshow(
            a[cls_token_count:, cls_token_count:],
            labels=dict(
                x="K",
                y="Q",
            ),
        )
        fig_mask.update_layout(
            template='plotly_dark',
            margin=dict(t=0,b=0,l=0,r=0),
            height=plot_size[0],
            width=plot_size[1],
        )
        fig_mask.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        # fig_mask.update_xaxes(tick0=0, dtick=1).update_yaxes(tick0=0, dtick=1)
        
        return fig_mask, fig_grid
    
    @classmethod
    def plot_and_save(cls,
                masks,
                dp='./plots',
                mask_type='dist',
                mask_levels=3,
                mask_levels_final=3,
                cls_token_type='sum',
                cls_token_pos=None,
                token_grid_size=14,
                cls_token_count=1,
                grid_sep=2,
                plot_size=[800, 800],
                ):
        
        fig_mask, fig_grid = cls.plot_masks(
            masks,
            cls_token_count=cls_token_count,
            token_grid_size=token_grid_size,
            # mask_levels=mask_levels,
            mask_levels_final=mask_levels_final,
            grid_sep=grid_sep,
            plot_size=plot_size,
        )
        _cls_str = cls_token_type
        if cls_token_type == 'pos':
            assert isinstance(cls_token_pos, (int, float))
            assert 0 <= cls_token_pos <= 1
            _cls_str = f'pos{cls_token_pos:.1f}'
        
        if not os.path.isdir(dp):
            os.makedirs(dp)
        fig_grid.write_image(os.path.join(dp, f'mask_grid_{mask_type}_hl{mask_levels}.png'))
        fig_mask.write_image(os.path.join(dp, f'mask_{mask_type}_hl{mask_levels}_{_cls_str}.png'))
        

# %%
class FishPP_Head:
    @classmethod
    def get_dims(cls, heads, global_heads, head_dim):
        # qkv_dim is normally [3 * heads * head_dim]
        qk_dim = global_heads * head_dim
        qkv_heads = heads + 2 * global_heads
        qkv_dim = (heads + 2 * global_heads) * head_dim
        v_dim = heads * head_dim
        return {
            'qkv_heads': qkv_heads,
            'qkv_dim': qkv_dim,
            'qk_dim': qk_dim,
            'v_dim': v_dim,
        }

# %%
def get_layer_indices(limit=None, offset=0, count=12):
    if limit is None or limit < 0:
        return list(range(count))
    assert isinstance(limit, int)
    assert isinstance(offset, int)
    assert offset >= 0
    _max = min(limit + offset, count)
    _indices = list(range(offset, _max))
    return _indices

# TODO: update plotting for 1D Masks
# %%
# if __name__ == '__main__':
#     save_plots = True
    
#     _cls_token_count = 1
#     _token_grid_size = 14
#     _mask_levels = 4
    
#     dp = f'../plots/t{_token_grid_size}_hl{_mask_levels}'
#     if save_plots and not os.path.isdir(dp):
#         os.makedirs(dp)
    
#     for _mask_type in ['dist', 'distq', 'cross', 'crossq']:
#         _cls_token_type = 'pos'
#         _cls_token_pos = 0.5
        
#         if _mask_type in ['h', 'hdist']:
#             _cls_token_type = 'sum'
#             _cls_token_pos = None
        
#         masks, mask_base, mask_levels_final = H_Matrix_Masks_1D.get_masks(
#             mask_type=_mask_type,
#             cls_token_type=_cls_token_type,
#             cls_token_pos=_cls_token_pos,
#             cls_token_count=_cls_token_count,
#             token_grid_size=_token_grid_size,
#             mask_levels=_mask_levels,
#         )
        
#         figs = H_Matrix_Masks_1D.plot_masks(
#             masks,
#             cls_token_count=_cls_token_count,
#             token_grid_size=_token_grid_size,
#             mask_levels_final=mask_levels_final,
#             grid_sep=2,
#             plot_size=[800, 800],
#         )
#         _ = [_fig.show() for _fig in figs]
        
#         # continue
#         if save_plots:
#             # _cls_str = _cls_token_type
#             # if _cls_token_type == 'pos':
#             #     _cls_str = f'pos{_cls_token_pos:.1f}'
            
#             # figs[0].write_image(os.path.join(dp, f'mask_{_mask_type}_hl{_mask_levels}_{_cls_str}.png'))
#             figs[0].write_image(os.path.join(dp, f'mask_{_mask_type}_hl{_mask_levels}.png'))
#             figs[1].write_image(os.path.join(dp, f'mask_grid_{_mask_type}_hl{_mask_levels}.png'))



# %%





