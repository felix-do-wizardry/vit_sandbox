# %%
import numpy as np
import json, os, time

# %%
# import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# %%
class H_Matrix:
    '''2D H_Matrix for Transformer Attention
    '''
    def __init__(self,
                t=14,
                level=2,
                mask_type='h',
                cls_token_count=1,
                cls_token_pos=0.5,
                cls_token_order='first',
                **kwargs,
                ):
        '''Args:
        t (int): number of token columns/rows, assuming square input
        level (int): level of h matrix
        '''
        
        assert mask_type in ['dist', 'hdist', 'dist', 'distq']
        self.mask_type = mask_type
        self.is_dist = mask_type in ['dist', 'distq']
        
        assert cls_token_order in ['first']
        
        self.cls_token_pad = None
        if cls_token_count < 1:
            self.cls_token_count = 0
            self.cls_token_order = None
        else:
            assert isinstance(cls_token_count, int)
            self.cls_token_count = cls_token_count
            assert cls_token_order in ['first', 'last'], f'`cls_token_order`=[{cls_token_order}]'
            self.cls_token_order = cls_token_order
            if cls_token_order == 'first':
                self.cls_token_pad = [[self.cls_token_count, 0]]
            else:
                self.cls_token_pad = [[0, self.cls_token_count]]
        
        if cls_token_pos is None or self.cls_token_count <= 0:
            self.cls_token_pos = None
        else:
            if isinstance(cls_token_pos, (float, int)):
                cls_token_pos = [cls_token_pos, cls_token_pos]
            assert isinstance(cls_token_pos, (list, tuple)), f'`cls_token_pos`=[{str(cls_token_pos)[:100]}] is not of type [list, tuple, float, int]'
            assert len(cls_token_pos) == 2, f'`cls_token_pos` has len [{len(cls_token_pos)}] != 2'
            assert all([0 <= v <= 1 for v in cls_token_pos]), f'`cls_token_pos`=[{str(cls_token_pos)[:100]}] must be in range [0, 1]'
            self.cls_token_pos = list(cls_token_pos)
        
        assert isinstance(t, int)
        assert t > 0
        self.t = t
        n_img = t * t
        n = n_img + self.cls_token_count
        self.n_img = n_img
        self.n = n
        
        assert isinstance(level, int)
        assert level > 0
        self.level = level
        w = t / 2 ** (level / 2)
        if not self.is_dist:
            assert level % 2 == 0
            assert w % 1 == 0
        
        self.w = int(w)
        self.cw = int(w * w)
        # w: window/cell size | cw: number of tokens per window/cell
        
        # m: h matrix mask
        self.m = np.zeros([self.n, self.n], dtype=int)
        for i in range(level):
            j = i + 1
            _s = int(n // 2 ** j)
            # print(i, j, _s, list(range(0, n, _s)))
            for r0 in range(0, n, _s):
                r1 = r0 + _s
                self.m[r0 : r1, r0 : r1] = j
        
        y = np.tile(np.arange(self.t).reshape(-1, 1), (1, self.t))
        x = np.tile(np.arange(self.t).reshape(1, -1), (self.t, 1))
        c = np.stack([y, x], -1)
        # c.shape
        
        cy = (np.floor(y / w)).astype(int)
        cx = (np.floor(x / w)).astype(int)
        # cy, cx
        
        cxf = cx.reshape(-1)
        cyf = cy.reshape(-1)
        # cxf == cyf
        
        match   = np.zeros([n_img, n_img], dtype=int) - 1
        match_h = np.zeros([n_img, n_img], dtype=int) - 1
        dist    = np.zeros([n_img, n_img], dtype=float)
        
        for iq in range(n_img):
            pq = (int(iq // t), iq % t)
            
            for ik in range(n_img):
                pk = (int(ik // t), ik % t)
                
                dist[iq, ik] = np.sqrt(np.sum((np.array(pq) - np.array(pk)) ** 2)) / t
                
                x_match = int(cxf[iq] == cxf[ik])
                y_match = int(cyf[iq] == cyf[ik])
                
                _match = x_match + y_match
                _match_h = x_match * y_match + y_match
                
                assert _match in [0, 1, 2]
                
                match[iq, ik] = _match
                match_h[iq, ik] = _match_h
        
        assert np.all(match >= 0), f'?? {np.min(match)}'
        assert np.all(match_h >= 0), f'?? {np.min(match_h)}'
        
        if self.cls_token_count > 0:
            # CLS
            match = np.pad(match, pad_width=self.cls_token_pad, constant_values=-1)
            match_h = np.pad(match_h, pad_width=self.cls_token_pad, constant_values=-1)
            dist = np.pad(dist, pad_width=self.cls_token_pad, constant_values=0.)
            
            if self.is_dist and self.cls_token_pos is not None:
                p_cls = np.array(self.cls_token_pos).astype(float) * (t - 1)
                for i in range(n_img):
                    p = np.array([int(i // t), i % t]).astype(float)
                    # CLS q
                    dist[0, i + self.cls_token_count] = np.sqrt(np.sum((p_cls - p) ** 2)) / t
                    # CLS k
                    dist[i + self.cls_token_count, 0] = np.sqrt(np.sum((p_cls - p) ** 2)) / t
                # CLS qk
                dist[0, 0] = 0.
        
        dist_neg = -dist
        digitize_levels = level + 1
        dist_dig = np.clip(np.digitize(
            dist_neg,
            np.percentile(
                dist_neg,
                np.linspace(0, 100, digitize_levels + 1),
            ),
        ), 1, digitize_levels) - 1
        
        _dist_pct = np.percentile(
            dist_neg,
            np.linspace(0, 100, digitize_levels + 1),
            axis=1,
        )

        distq_dig = np.clip([
            np.digitize(
                dist_neg[i],
                _dist_pct[:, i],
            )
            for i in range(dist.shape[0])
        ], 1, digitize_levels) - 1

        # CLS
        if self.cls_token_count > 0:
            # CLS
            if self.is_dist and self.cls_token_pos is None:
                dist_dig[:self.cls_token_count, :] = -1
                dist_dig[:, :self.cls_token_count] = -1
                distq_dig[:self.cls_token_count, :] = -1
                distq_dig[:, :self.cls_token_count] = -1
                
        
        # shape = [t*t+1, t*t+1], range = [-1, level]
        self.match = match
        self.match_h = match_h
        self.dist = dist
        self.dist_dig = dist_dig
        self.distq_dig = distq_dig
        
        
        if mask_type in ['h']:
            self.indexed_mask = self.match_h
        elif mask_type in ['hdist']:
            self.indexed_mask = self.match
        elif mask_type in ['dist']:
            self.indexed_mask = self.dist_dig
        elif mask_type in ['distq']:
            self.indexed_mask = self.distq_dig
        else:
            raise NotImplementedError(f'`mask_type`[{mask_type}] has not been implemented')


# %%
class H_Matrix_Masks:
    def __init__(self,
                mask_type='distq',
                cls_token_type='pos',
                cls_token_pos=0.5,
                cls_token_count=1,
                token_grid_size=14,
                mask_levels=3,
                ):
        self.masks, self.mask_base, self.mask_levels_final = self.get_masks(
            mask_type=mask_type,
            cls_token_type=cls_token_type,
            cls_token_pos=cls_token_pos,
            cls_token_count=cls_token_count,
            token_grid_size=token_grid_size,
            mask_levels=mask_levels,
        )
    
    @classmethod
    def get_masks(cls,
                mask_type='distq',
                cls_token_type='pos',
                cls_token_pos=0.5,
                cls_token_count=1,
                token_grid_size=14,
                mask_levels=3,
                ):
        
        # mask_type = 'hdist'
        # cls_token_type = 'sum'
        # cls_token_pos = None

        # mask_type = 'dist'
        # mask_type = 'distq'
        # cls_token_type = 'pos'
        # cls_token_pos = 0.5
        
        assert mask_type in ['h', 'hdist', 'dist', 'distq']
        assert cls_token_type in ['copy', 'pos', 'mask', 'sum']
        if cls_token_type == 'pos':
            assert isinstance(cls_token_pos, (float, int))
            assert 1 >= cls_token_pos >= 0
        else:
            cls_token_pos = None

        # mask_levels = 3
        token_count = int(token_grid_size ** 2) + cls_token_count

        hm = H_Matrix(
            t=token_grid_size,
            level=mask_levels - 1,
            mask_type=mask_type,
            cls_token_count=cls_token_count,
            cls_token_order='first',
            cls_token_pos=cls_token_pos,
        )

        indexed_mask = hm.indexed_mask
        mask_levels_final = mask_levels

        # [N, N] -> [N, N] -> [1, 1, N, N, 1]
        mask_base = np.array(
            indexed_mask * 0,
            dtype=np.float32,
        )[None, None, :, :, None]

        _shape = list(indexed_mask.shape)
        assert (len(_shape) == 2 and _shape[0] == _shape[1] == token_count
            ), f'indexed_mask.shape={_shape} != token_count[{token_count}]'
        
        
        if cls_token_type in ['copy']:
            # [copy] not using cls token projection -> set mask_base to 1 for -1 values (cls)
            assert np.all(indexed_mask[:cls_token_count, :] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[:, :cls_token_count] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[cls_token_count:, cls_token_count:] >= 0), '?? non-cls mask (should be >= 0)'
            
            # [N, N] -> [N, N] -> [1, 1, N, N, 1]
            mask_base = np.array(
                indexed_mask == -1,
                dtype=np.float32,
            )[None, None, :, :, None]
            
            # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
            masks = np.array([
                    indexed_mask == i
                    for i in range(mask_levels)
                ],
                dtype=np.float32,
            ).transpose(1, 2, 0)[None, None]
            
        elif cls_token_type in ['pos']:
            assert np.all(indexed_mask >= 0), 'dist pos ?? cls mask (should be >= 0)'
            
            # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
            masks = np.array([
                    indexed_mask == i
                    for i in range(mask_levels)
                ],
                dtype=np.float32,
            ).transpose(1, 2, 0)[None, None]

        elif cls_token_type in ['mask']:
            # [mask] using cls token projection -> set masks as new value for -1 in indexed_mask
            assert np.all(indexed_mask[:cls_token_count, :] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[:, :cls_token_count] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[cls_token_count:, cls_token_count:] >= 0), '?? non-cls mask (should be >= 0)'
            
            # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
            masks = np.array([
                    indexed_mask == i
                    for i in [*list(range(mask_levels)), -1]
                ],
                dtype=np.float32,
            ).transpose(1, 2, 0)[None, None]
            mask_levels_final = mask_levels + 1
            
        elif cls_token_type in ['sum']:
            # [sum] using cls token projection -> set masks to all True for -1 in indexed_mask
            assert np.all(indexed_mask[:cls_token_count, :] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[:, :cls_token_count] == -1), '?? cls mask (should be -1)'
            assert np.all(indexed_mask[cls_token_count:, cls_token_count:] >= 0), '?? non-cls mask (should be >= 0)'
            
            # [Level, N, N] -> [N, N, Level] -> [1, 1, N, N, Level]
            masks = np.array([
                    np.any([indexed_mask == i, indexed_mask == -1], axis=0)
                    for i in range(mask_levels)
                ],
                dtype=np.float32,
            ).transpose(1, 2, 0)[None, None]
        
        print(f'type[{mask_type}] global[{"?"}] cls[{cls_token_type}] mask_levels_final[{mask_levels_final}]')
        print(f'masks{list(masks.shape)} mask_base{list(mask_base.shape)}')
        # masks.shape, mask_base.shape
        
        return masks, mask_base, mask_levels_final
    
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
# _mask_type='distq'
# _cls_token_type='pos'
# _cls_token_pos=0.5

# _mask_type='dist'
# _cls_token_type='pos'
# _cls_token_pos=0.5

# # _mask_type='hdist'
# # _cls_token_type='sum'
# # _cls_token_pos=None


# _cls_token_count=0
# _token_grid_size=7
# _mask_levels=5

# masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks(
#     mask_type=_mask_type,
#     cls_token_type=_cls_token_type,
#     cls_token_pos=_cls_token_pos,
#     cls_token_count=_cls_token_count,
#     token_grid_size=_token_grid_size,
#     mask_levels=_mask_levels,
# )

# figs = H_Matrix_Masks.plot_masks(
#     masks,
#     cls_token_count=_cls_token_count,
#     token_grid_size=_token_grid_size,
#     mask_levels_final=mask_levels_final,
#     grid_sep=2,
#     plot_size=[800, 800],
# )
# _ = [_fig.show() for _fig in figs]

# %%









# # %%
# t = 14
# a = masks.reshape(token_count, token_count, mask_levels_final)
# a = np.sum(a * np.arange(mask_levels_final).reshape(1, 1, -1), -1)
# b = a[1:, 1:].reshape(t, t, t, t)
# _sep = 2
# _img = np.full([t * (t + _sep) - _sep] * 2, fill_value=None)

# # _ticks = [' ' for i in range(_img.shape[0])]
# for qy in range(t):
#     # _ticks[qy * (t + _sep) + int(t // 2)] = str(qy)
#     for qx in range(t):
#         _y = qy * (t + _sep)
#         _x = qx * (t + _sep)
#         _img[
#             _y : _y + t,
#             _x : _x + t,
#         ] = b[qy, qx]

# # b = a.transpose(0, 2, 1, 3)
# # b = b.reshape(t*t, t*t)
# a.shape, b.shape, _img.shape

# # %%
# fig = px.imshow(
#     # b,
#     _img,
#     labels=dict(
#         x="Q column",
#         y="Q row",
#         # color="Mask Index",
#     ),
#     y=(np.arange(_img.shape[0]) - t/2 - 0.5) / (t + _sep),
#     x=(np.arange(_img.shape[1]) - t/2 - 0.5) / (t + _sep),
# )
# fig.update_layout(
#     template='plotly_dark',
#     margin=dict(t=0,b=0,l=0,r=0),
#     height=800,
#     width=800,
# )
# # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.update_xaxes(tick0=0, dtick=1).update_yaxes(tick0=0, dtick=1)

# _cls_str = cls_token_type
# if cls_token_type == 'pos':
#     _cls_str = f'pos{cls_token_pos:.1f}'

# fig.write_image(f'./plots/mask_grid_{mask_type}_hl{mask_levels}.png')
# fig


# # %%
# fig = px.imshow(
#     a[1:, 1:],
#     labels=dict(
#         x="K",
#         y="Q",
#     ),
# )
# fig.update_layout(
#     template='plotly_dark',
#     margin=dict(t=0,b=0,l=0,r=0),
#     height=800,
#     width=800,
# )
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# # fig.update_xaxes(tick0=0, dtick=1).update_yaxes(tick0=0, dtick=1)

# _cls_str = cls_token_type
# if cls_token_type == 'pos':
#     _cls_str = f'pos{cls_token_pos:.1f}'

# fig.write_image(f'./plots/mask_{mask_type}_hl{mask_levels}_{_cls_str}.png')
# fig

# %%





