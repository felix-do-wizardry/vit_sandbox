# %%
import numpy as np
import json, os, time

# import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# %%
# swin
MASK_TYPES = [
    # 'h', 'hdist',
    'dist', 'distq',
    'cross',
    # 'crossq',
    # 'col', 'row',
    # 'x',
]

# deit
# MASK_TYPES = [
#     'h', 'hdist',
#     'dist', 'distq',
#     'cross',
#     'crossq',
# ]

MASK_TYPES_DIST = [
    'dist', 'distq',
    'cross',
    # 'crossq',
]

CLS_TOKEN_TYPES = [
    'copy',
    'pos',
    'mask',
    'sum',
]

# %%
class Matrix:
    @classmethod
    def multi_range(cls, _max=4, count=2, flat=True):
        assert isinstance(count, int) and count > 0
        indices = np.arange(_max ** count)
        _mr = np.floor((indices[None] / _max ** np.arange(count).reshape(-1, 1)[::-1]) % _max).astype(int)
        _mr = _mr.T
        if flat:
            return _mr
        else:
            return _mr.reshape([_max] * count + [count])
    
    @classmethod
    def get_grid_pos(cls, h=4, w=None):
        assert isinstance(h, int) and h > 0
        if w is None:
            return cls.multi_range(h, count=2, flat=False)
        assert isinstance(w, int) and w > 0
        mr = cls.multi_range(max(h, w), count=2, flat=False).reshape(h, w, 2)
        return mr[:h, :w]
    
    @classmethod
    def plot_matrix_grid(cls, z, sep=1, size=(800, 1000), with_axis=True):
        if len(z.shape) == 2:
            # 2D array -> assume square grid + square cell
            assert np.all(np.sqrt(z.shape) % 1 == 0)
            _s0 = np.sqrt(z.shape).astype(int)
            z = z.reshape(_s0[0], _s0[0], _s0[1], _s0[1])
        assert len(z.shape) == 4
        assert isinstance(sep, int) and sep >= 0
        z2 = np.pad(
            z.astype(float),
            [[0, 0], [0, 0], [0, sep], [0, sep]],
            constant_values=None,
        )
        z3 = z2.transpose(0, 2, 1, 3)
        z3 = z3.reshape(np.prod(z3.shape[0:2]), np.prod(z3.shape[2:4]))
        z4 = z3[:-sep, :-sep]
        
        if with_axis:
            _s = z.shape
            x_axis_values = np.linspace(0.5, _s[1] + 0.5, (_s[3] + 1) * _s[1] + 1)[1:-1]
            y_axis_values = np.linspace(0.5, _s[0] + 0.5, (_s[2] + 1) * _s[0] + 1)[1:-1]
        else:
            x_axis_values = None
            y_axis_values = None
        
        fig = px.imshow(
            img=z4,
            x=x_axis_values,
            y=y_axis_values,
        )
        fig.update_layout(
            margin=dict(t=0,b=0,l=0,r=0),
            template='plotly_dark',
            height=size[0],
            width=size[1],
        )
        # fig = cls.plot_matrix(
        #     z=z4,
        #     size=size,
        #     x=x_axis_values,
        #     y=y_axis_values,
        # )
        return fig
    
    @classmethod
    def plot_matrix(cls, z, size=(800, 1000), x=None, y=None):
        assert len(z.shape) in [2]
        fig = px.imshow(
            z,
            x=x,
            y=y,
        )
        fig.update_layout(
            margin=dict(t=0,b=0,l=0,r=0),
            template='plotly_dark',
            height=size[0],
            width=size[1],
        )
        return fig
    
    @classmethod
    def digitize(cls, a, a_values=None, levels=3):
        '''digitize across all the axis
        '''
        if a_values is None:
            a_values = a
        _dig = np.clip(np.digitize(
            a,
            np.percentile(
                a_values,
                np.linspace(0, 100, levels + 1),
            ),
        ), 1, levels) - 1
        return _dig
    
    @classmethod
    def digitize_axis(cls, a, a_values=None, levels=3):
        '''digitize across the 2nd axis (of the 2 axes)
        '''
        if a_values is None:
            a_values = a
        assert len(a.shape) == 2
        _pct = np.percentile(
            a_values,
            np.linspace(0, 100, levels + 1),
            axis=1,
        )
        _dig = np.clip([
            np.digitize(
                a[i],
                _pct[:, i],
            )
            for i in range(a.shape[0])
        ], 1, levels) - 1
        _dig.shape
        return _dig
    
    @classmethod
    def get_cross_matrix(cls, t=7, levels=3):
        assert isinstance(t, int) and t >= 2
        assert isinstance(levels, int) and levels >= 2
        assert t >= levels
        
        g = cls.get_grid_pos(t)
        
        cross_qk = np.min(np.abs(g[:, :, None, None] - g[None, None]), -1)
        cross_qk.shape
        # cls.plot_matrix_grid(cross_qk, 1, [600, 640]).show()
        
        bins_mr = cls.multi_range(t - 1, count=levels-1) + 1
        # bins_mr.shape
        bins_sets = bins_mr[np.all(bins_mr[:, 1:] > bins_mr[:, :-1], axis=-1)]
        # bins_sets.shape
        data = []
        for _bins in bins_sets:
            _cross_dig = np.digitize(
                cross_qk,
                bins=_bins,
            )
            _hist = np.histogram(_cross_dig, bins=levels)[0]
            _hist_r = _hist / np.prod(_cross_dig.shape)
            _std = np.std(_hist_r)
            data.append({
                'bins': list(_bins.astype(int)),
                'hist': _hist,
                'hist_pct': (_hist_r * 100).round(0).astype(int),
                'std': _std,
            })
        assert len(data) > 0
        
        df = pd.DataFrame(data)
        # df.sort_values(['std'])
        
        _bin_best = df.sort_values(['std'])['bins'][0]
        cross_dig_qk = np.digitize(
            cross_qk,
            bins=_bin_best,
        )
        # cls.plot_matrix_grid(cross_dig_qk, 1, [600, 640]).show()
        # cross_dig_qk.shape
        
        cross_dig = cross_dig_qk.reshape(t*t, t*t)
        return cross_dig
    
    @classmethod
    def get_dist_matrix(cls, t=7, levels=3):
        assert isinstance(t, int) and t >= 2
        mr = cls.multi_range(t, 4, flat=False)
        # mr.shape
        dist_yx_qk = mr[..., :2] - mr[..., 2:]
        # dist_yx_qk.shape
        dist_qk = np.sqrt((dist_yx_qk ** 2).sum(-1))
        # dist_qk.shape
        
        # cls.plot_matrix_grid(dist_qk, 1, [600, 640])
        
        dist = dist_qk.reshape(t*t, t*t)
        
        dist_dig = cls.digitize(-dist, levels=levels)
        distq_dig = cls.digitize_axis(-dist, levels=levels)
        # cls.plot_matrix_grid(dist_dig, 1, [600, 640]).show()
        # cls.plot_matrix_grid(distq_dig, 1, [600, 640]).show()
        return dist_dig, distq_dig
    
    @classmethod
    def get_dist_matrix_cls(cls, t=7, levels=3, cls_token_count=1, cls_token_pos=None):
        # CLS tokens are added at the beginning of the array
        if cls_token_pos is None:
            cls_token_using_pos = False
            cls_token_pos = 0.5
        else:
            cls_token_using_pos = True
        assert 0 <= cls_token_pos <= 1
        mr = Matrix.multi_range(t, 2, flat=False).reshape(t*t, 2)
        mr = np.concatenate([
            [[(t - 1) * cls_token_pos] * 2] * cls_token_count,
            mr,
        ], axis=0).astype(float)
        # mr.shape
        
        dist_yx = mr[None] - mr[:, None]
        # dist_yx.shape
        dist = np.sqrt((dist_yx ** 2).sum(-1))
        # dist
        
        # Matrix.plot_matrix(dist, (600, 640))
        
        dist_dig = Matrix.digitize(-dist, -dist[cls_token_count:, cls_token_count:], levels=levels)
        distq_dig = Matrix.digitize_axis(-dist, -dist[:, cls_token_count:], levels=levels)
        # Matrix.plot_matrix_grid(dist_dig[1:, 1:], 1, [600, 640]).show()
        # Matrix.plot_matrix_grid(distq_dig[1:, 1:], 1, [600, 640]).show()
        
        if not cls_token_using_pos and cls_token_count > 0:
            # set all CLS values to -1
            dist_dig[:cls_token_count, :] = -1
            distq_dig[:cls_token_count, :] = -1
            dist_dig[:, :cls_token_count] = -1
            distq_dig[:, :cls_token_count] = -1
        
        # dist_dig.shape, distq_dig.shape
        return dist_dig, distq_dig


# %%
# # %%
# t = 14
# levels = 4
# # CLS tokens are added at the beginning of the array
# mr = Matrix.multi_range(t, 2, flat=False).reshape(t*t, 2)
# mr = np.concatenate([
#     [[t / 2 - .5] * 2],
#     mr,
# ], axis=0).astype(float)
# mr.shape

# dist_yx = mr[None] - mr[:, None]
# dist_yx.shape
# dist = np.sqrt((dist_yx ** 2).sum(-1))
# dist

# # Matrix.plot_matrix(dist, (600, 640))

# dist_dig = Matrix.digitize(-dist, -dist[1:, 1:], levels=levels)
# distq_dig = Matrix.digitize_axis(-dist, -dist[:, 1:], levels=levels)
# # Matrix.plot_matrix_grid(dist_dig[1:, 1:], 1, [600, 640]).show()
# # Matrix.plot_matrix_grid(distq_dig[1:, 1:], 1, [600, 640]).show()

# dist_dig.shape, distq_dig.shape
# # return dist_dig, distq_dig

# %%
# t = 7
# levels = 3
# dist_dig, distq_dig = Matrix.get_dist_matrix(t=t, levels=levels)
# print('> dist')
# Matrix.plot_matrix_grid(dist_dig, 1, [600, 640]).show()
# print('> distq')
# Matrix.plot_matrix_grid(distq_dig, 1, [600, 640]).show()

# cross_dig = Matrix.get_cross_matrix(t=t, levels=levels)
# print('> crossq')
# Matrix.plot_matrix_grid(cross_dig, 1, [600, 640]).show()

# # %%
# t = 7
# levels = 3
# cls_token_count = 1
# cls_token_pos = 0.5
# dist_dig, distq_dig = Matrix.get_dist_matrix_cls(
#     t=t,
#     levels=levels,
#     cls_token_count=cls_token_count,
#     cls_token_pos=cls_token_pos,
# )
# print('> dist')
# Matrix.plot_matrix(dist_dig, [600, 640]).show()
# print('> distq')
# Matrix.plot_matrix(distq_dig, [600, 640]).show()

# %%
class H_Matrix_New:
    def __init__(self,
                t=14,
                levels=2,
                mask_type='dist',
                cls_token_count=0,
                cls_token_pos=0.5,
                # cls_token_order='first',
                **kwargs,
                ):
        '''Args:
            t (int): number of token columns/rows, assuming square input
            level (int): level of h matrix
        '''
        assert mask_type in MASK_TYPES, f''
        print(f'<H_Matrix_New mask_type[{mask_type}] t[{t}] levels[{levels}] cls_token_count[{cls_token_count}] cls_token_pos[{cls_token_pos}]>')
        
        # cross currently does NOT support cls_token
        cross_dig = Matrix.get_cross_matrix(t=t, levels=levels)
        # print('> crossq')
        # Matrix.plot_matrix_grid(cross_dig, 1, [600, 640]).show()

        if cls_token_count <= 0:
            dist_dig, distq_dig = Matrix.get_dist_matrix(
                t=t,
                levels=levels,
            )
            # print('> dist')
            # Matrix.plot_matrix_grid(dist_dig, 1, [600, 640]).show()
            # print('> distq')
            # Matrix.plot_matrix_grid(distq_dig, 1, [600, 640]).show()
        else:
            # cls_token_pos = 0.5
            dist_dig, distq_dig = Matrix.get_dist_matrix_cls(
                t=t,
                levels=levels,
                cls_token_count=cls_token_count,
                cls_token_pos=cls_token_pos,
            )
            # print('> dist (flat)')
            # Matrix.plot_matrix(dist_dig, [600, 640]).show()
            # print('> distq (flat)')
            # Matrix.plot_matrix(distq_dig, [600, 640]).show()
        
        self.cross_dig = cross_dig
        self.dist_dig = dist_dig
        self.distq_dig = distq_dig
        
        if mask_type == 'dist':
            self.indexed_mask = dist_dig
        elif mask_type == 'distq':
            self.indexed_mask = distq_dig
        elif mask_type == 'cross':
            self.indexed_mask = cross_dig
            assert cls_token_count <= 0, f'mask_type[{mask_type}] currently does not support CLS token'
        else:
            raise ValueError(f'mask_type[{mask_type}] is not available!')

        

# %%
def analyze_cross_bins(cross_qk, df_bins, t, levels, top_k=5):
    _bin_bests = df_bins.sort_values(['std'])['bins'][:top_k]
    data2 = []
    for _bin in _bin_bests[::-1]:
        _cross_dig = np.digitize(
            cross_qk,
            bins=_bin,
            # bins=[1, 2, 4],
        )
        _hist2d = (_cross_dig.reshape(t*t,t*t,1) == np.arange(levels)).mean(1)
        _hist2d.shape
        _std2d = np.std(_hist2d, axis=-1)
        data2.extend([
            {
                'q': i / max(_std2d.shape[0] - 1, 1),
                'std': v,
                'bin': tuple(_bin),
            }
            for i, v in enumerate(sorted(_std2d))
            # for i, v in enumerate(_std2d)
        ])
    if len(data2) == 0:
        return None
    _df = pd.DataFrame(data2)
    fig = px.line(
        _df,
        x='q',
        y='std',
        color='bin',
        template='plotly_dark',
        range_y=[0, max(_df['std']) * 1.1],
        # range_x=[0, 1],
    )
    return fig

def analyze_dist(dist_dig, distq_dig, levels, t):
    _std = np.sort((dist_dig.reshape(t*t, t*t, 1) == np.arange(levels)).mean(1).std(-1))
    _std_q = np.sort((distq_dig.reshape(t*t, t*t, 1) == np.arange(levels)).mean(1).std(-1))
    _df = pd.DataFrame([
        {
            'q': i / max(1, __std.shape[0] - 1),
            'std': v,
            'mask_type': _type,
        }
        for _type, __std in zip(['dist', 'distq'], [_std, _std_q])
        for i, v in enumerate(__std)
    ])
    fig = px.line(
        _df,
        x='q',
        y='std',
        color='mask_type',
        template='plotly_dark',
        range_y=[0, np.max(_df['std']) * 1.1],
    )
    return fig


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
                # cls_token_order='first',
                **kwargs,
                ):
        '''Args:
        t (int): number of token columns/rows, assuming square input
        level (int): level of h matrix
        '''
        
        assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        self.mask_type = mask_type
        self.is_dist = mask_type in MASK_TYPES_DIST
        
        cls_token_order = 'first'
        # assert cls_token_order in ['first']
        
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
        x_dist  = np.zeros([n_img, n_img], dtype=float)
        y_dist  = np.zeros([n_img, n_img], dtype=float)
        
        for iq in range(n_img):
            pq = np.array([int(iq // t), iq % t], dtype=float)
            
            for ik in range(n_img):
                pk = np.array([int(ik // t), ik % t], dtype=float)
                
                dist[iq, ik] = np.sqrt(np.sum((pq - pk) ** 2)) / t
                y_dist[iq, ik] = np.abs(pq[0] - pk[0])
                x_dist[iq, ik] = np.abs(pq[1] - pk[1])
                
                
                x_match = int(cxf[iq] == cxf[ik])
                y_match = int(cyf[iq] == cyf[ik])
                
                _match = x_match + y_match
                _match_h = x_match * y_match + y_match
                
                assert _match in [0, 1, 2]
                
                match[iq, ik] = _match
                match_h[iq, ik] = _match_h
        
        assert np.all(match >= 0), f'?? {np.min(match)}'
        assert np.all(match_h >= 0), f'?? {np.min(match_h)}'
        
        # padding for CLS tokens
        if self.cls_token_count > 0:
            # CLS
            match = np.pad(match, pad_width=self.cls_token_pad, constant_values=-1)
            match_h = np.pad(match_h, pad_width=self.cls_token_pad, constant_values=-1)
            dist = np.pad(dist, pad_width=self.cls_token_pad, constant_values=0.)
            y_dist = np.pad(y_dist, pad_width=self.cls_token_pad, constant_values=0)
            x_dist = np.pad(x_dist, pad_width=self.cls_token_pad, constant_values=0)
            
            if self.cls_token_pos is not None:
                p_cls = np.array(self.cls_token_pos).astype(float) * (t - 1)
                for i in range(n_img):
                    p = np.array([int(i // t), i % t]).astype(float)
                    
                    # CLS q
                    dist[:self.cls_token_count, i + self.cls_token_count] = np.sqrt(np.sum((p_cls - p) ** 2)) / t
                    y_dist[:self.cls_token_count, i + self.cls_token_count] = np.abs(p_cls[0] - p[0])
                    x_dist[:self.cls_token_count, i + self.cls_token_count] = np.abs(p_cls[1] - p[1])
                    
                    # CLS k
                    dist[i + self.cls_token_count, :self.cls_token_count] = np.sqrt(np.sum((p_cls - p) ** 2)) / t
                    y_dist[i + self.cls_token_count, :self.cls_token_count] = np.abs(p_cls[0] - p[0])
                    x_dist[i + self.cls_token_count, :self.cls_token_count] = np.abs(p_cls[1] - p[1])
                
                # CLS qk
                dist[:self.cls_token_count, :self.cls_token_count] = 0.
                y_dist[:self.cls_token_count, :self.cls_token_count] = 0.
                x_dist[:self.cls_token_count, :self.cls_token_count] = 0.
        
        cross_dist = np.min([y_dist, x_dist], axis=0)
        
        digitize_levels = level + 1
        
        dist_neg = -dist
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
        
        
        cross_dist_neg = -cross_dist
        cross_dist_dig = np.clip(np.digitize(
            cross_dist_neg,
            np.percentile(
                cross_dist_neg,
                np.linspace(0, 100, digitize_levels + 1),
            ),
        ), 1, digitize_levels) - 1
        
        _cross_dist_pct = np.percentile(
            cross_dist_neg,
            np.linspace(0, 100, digitize_levels + 1),
            axis=1,
        )

        cross_distq_dig = np.clip([
            np.digitize(
                cross_dist_neg[i],
                _cross_dist_pct[:, i],
            )
            for i in range(cross_dist.shape[0])
        ], 1, digitize_levels) - 1


        # CLS - if not pos - set all dist cls entries to -1
        if self.cls_token_count > 0 and self.cls_token_pos is None:
            # CLS q
            dist_dig[:self.cls_token_count, :] = -1
            distq_dig[:self.cls_token_count, :] = -1
            cross_dist_dig[:self.cls_token_count, :] = -1
            cross_distq_dig[:self.cls_token_count, :] = -1
            
            # CLS k
            dist_dig[:, :self.cls_token_count] = -1
            distq_dig[:, :self.cls_token_count] = -1
            cross_dist_dig[:, :self.cls_token_count] = -1
            cross_distq_dig[:, :self.cls_token_count] = -1
                
        
        # shape = [t*t+1, t*t+1], range = [-1, level]
        self.match = match
        self.match_h = match_h
        self.dist = dist
        self.dist_dig = dist_dig
        self.distq_dig = distq_dig
        self.cross_dist_dig = cross_dist_dig
        self.cross_distq_dig = cross_distq_dig
        
        
        if mask_type in ['h']:
            self.indexed_mask = self.match_h
        elif mask_type in ['hdist']:
            self.indexed_mask = self.match
        elif mask_type in ['dist']:
            self.indexed_mask = self.dist_dig
        elif mask_type in ['distq']:
            self.indexed_mask = self.distq_dig
        elif mask_type in ['cross']:
            self.indexed_mask = self.cross_dist_dig
        elif mask_type in ['crossq']:
            self.indexed_mask = self.cross_distq_dig
        else:
            raise NotImplementedError(f'`mask_type`[{mask_type}] has not been implemented')
    
    @classmethod
    def get_cross_bins(cls, t, level):
        '''
        level: the final level, or the value count
        '''
        assert isinstance(t, int)
        assert t >= 2
        # t 
        bins = []
        thresholds = []
        # for i in range(1, level):
        #     # _bin = [0, 1]
        #     # _thr = 
        #     thresholds.append()
        #     bins.append()
    
    @classmethod
    def get_dist(cls, p0, p1, t):
        return np.sqrt(np.sum((p0 - p1) ** 2)) / t
    

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
                cls_token_count=0,
                token_grid_size=14,
                mask_levels=3,
                ):
        
        assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        assert cls_token_type in CLS_TOKEN_TYPES, f'cls_token_type[{cls_token_type}] must be one of {CLS_TOKEN_TYPES}'
        if cls_token_type == 'pos':
            assert isinstance(cls_token_pos, (float, int))
            assert 1 >= cls_token_pos >= 0
        else:
            cls_token_pos = None
        
        token_count = int(token_grid_size ** 2) + cls_token_count
        
        # hm = H_Matrix(
        #     t=token_grid_size,
        #     level=mask_levels - 1,
        #     mask_type=mask_type,
        #     cls_token_count=cls_token_count,
        #     cls_token_order='first',
        #     cls_token_pos=cls_token_pos,
        # )
        hm = H_Matrix_New(
            t=token_grid_size,
            levels=mask_levels,
            mask_type=mask_type,
            cls_token_count=cls_token_count,
            cls_token_pos=cls_token_pos,
        )
        
        indexed_mask = hm.indexed_mask
        mask_levels_final = mask_levels

        # [N, N] -> [N, N] -> [1, 1, N, N, 1]
        mask_base = np.array(
            indexed_mask * 0,
            dtype=np.float32,
        )[None, None, :, :, None]
        
        # masks = np.array([
        #         indexed_mask == i
        #         for i in range(mask_levels)
        #     ],
        #     dtype=np.float32,
        # ).transpose(1, 2, 0)[None, None]
        
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
if __name__ == '__main__':
    save_plots = False
    
    _cls_token_count = 0
    _token_grid_size = 7
    _mask_levels = 4
    
    dp = f'./plots/t{_token_grid_size}_hl{_mask_levels}'
    if save_plots and not os.path.isdir(dp):
        os.makedirs(dp)
    
    for _mask_type in ['dist', 'distq', 'cross']:
        _cls_token_type = 'pos'
        _cls_token_pos = 0.5
        
        if _mask_type in ['h', 'hdist']:
            _cls_token_type = 'sum'
            _cls_token_pos = None
        
        masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks(
            mask_type=_mask_type,
            cls_token_type=_cls_token_type,
            cls_token_pos=_cls_token_pos,
            cls_token_count=_cls_token_count,
            token_grid_size=_token_grid_size,
            mask_levels=_mask_levels,
        )
        
        figs = H_Matrix_Masks.plot_masks(
            masks,
            cls_token_count=_cls_token_count,
            token_grid_size=_token_grid_size,
            mask_levels_final=mask_levels_final,
            grid_sep=1,
            plot_size=[800, 800],
        )
        _ = [_fig.show() for _fig in figs]
        
        # continue
        if save_plots:
            _cls_str = _cls_token_type
            if _cls_token_type == 'pos':
                _cls_str = f'pos{_cls_token_pos:.1f}'
            
            figs[0].write_image(os.path.join(dp, f'mask_{_mask_type}_hl{_mask_levels}_{_cls_str}.png'))
            figs[1].write_image(os.path.join(dp, f'mask_grid_{_mask_type}_hl{_mask_levels}.png'))


# %%





