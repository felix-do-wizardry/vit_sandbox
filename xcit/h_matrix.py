# %%
import numpy as np
import json, os, time

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# %%
MASK_TYPES_DIST = [
    'dist', 'distq',
    'cross',
    'crossq',
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
    def plot_matrix_grid(cls, z, sep=1, size=(600, 660), with_axis=True):
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
    def plot_matrix(cls, z, size=(600, 660), x=None, y=None):
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
        
        # _bin_best = df.sort_values(['std'])['bins'][0]
        _bin_best = df.sort_values(['std']).to_dict('records')[0]['bins']
        
        
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
    
    @classmethod
    def get_factor_power(cls, v, f):
        p = (v % (f**(np.arange(int(np.log(v)//np.log(f)+1)) + 1)) == 0).sum()
        return int(p)
    
    @classmethod
    def get_h_matrix(cls, n=64, levels=3):
        # level_step == 2
        h_levels = levels - 1
        
        max_levels = cls.get_factor_power(n, 2)
        assert h_levels <= max_levels, f'levels[{levels} > {max_levels+1}] is too high for n[{n}]'
        
        mesh = np.stack(np.meshgrid(np.arange(n), np.arange(n))[::-1], axis=-1)
        
        ml = np.floor(mesh[..., None] / n * 2**np.arange(1, h_levels+1)).astype(int)
        ml2 = ml[:, :, 0] == ml[:, :, 1]
        ml3 = ml2.sum(-1).astype(int)
        # Matrix.plot_matrix(ml3)
        return ml3


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
    mask_types = [
        # 'h', 'hdist',
        'dist', 'distq',
        'cross',
        # 'crossq',
    ]
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
        assert mask_type in self.mask_types, f'mask_type[{mask_type}] must be one of {self.mask_types}'
        print(f'<H_Matrix mask_type[{mask_type}] t[{t}] levels[{levels}] cls_token_count[{cls_token_count}] cls_token_pos[{cls_token_pos}]>')
        
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
class H_Matrix_1D:
    mask_types = [
        'h',
    ]
    
    def __init__(self,
                # t=14,
                n=16,
                levels=3,
                mask_type='dist',
                cls_token_count=0,
                cls_token_pos=0.5,
                # TODO: code cls_token_order
                # cls_token_order='first',
                **kwargs,
                ):
        '''Args:
            t (int): number of token columns/rows, assuming square input
            level (int): level of h matrix
        '''
        assert mask_type in self.mask_types, f'mask_type[{mask_type}] must be one of {self.mask_types}'
        print(f'<H_Matrix_1D mask_type[{mask_type}] n[{n}] levels[{levels}] cls_token_count[{cls_token_count}] cls_token_pos[{cls_token_pos}]>')
        
        assert cls_token_count <= 0, f'H_Matrix_1D currently does not support CLS tokens'
        
        h_dig = Matrix.get_h_matrix(
            n=n,
            levels=levels,
        )
        # print('> h')
        # Matrix.plot_matrix(h_dig).show()
        
        self.h_dig = h_dig
        # self.cross_dig = cross_dig
        # self.dist_dig = dist_dig
        # self.distq_dig = distq_dig
        
        if mask_type == 'h':
            self.indexed_mask = h_dig
        else:
            raise ValueError(f'mask_type[{mask_type}] is not available!')


# %%
class H_Matrix_Masks:
    
    @classmethod
    def process_masks(cls,
                indexed_mask,
                mask_levels,
                cls_token_type='pos',
                cls_token_count=0,
                ):
        '''
        indexed_mask [N, N] values=int[-1, mask_levels)
        '''
        mask_levels_final = mask_levels
    
        # [N, N] -> [N, N] -> [1, 1, N, N, 1]
        mask_base = np.array(
            indexed_mask * 0,
            dtype=np.float32,
        )[None, None, :, :, None]
        
        # [1, 1, N, N, L]
        masks = np.array([
                indexed_mask == i
                for i in range(mask_levels)
            ],
            dtype=np.float32,
        ).transpose(1, 2, 0)[None, None]
        
        if cls_token_count > 0:
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
        
        return masks, mask_base, mask_levels_final
    
    @classmethod
    def get_masks(cls,
                mask_type='distq',
                cls_token_type='pos',
                cls_token_pos=0.5,
                cls_token_count=0,
                token_grid_size=14,
                mask_levels=3,
                ):
        
        # assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        assert cls_token_type in CLS_TOKEN_TYPES, f'cls_token_type[{cls_token_type}] must be one of {CLS_TOKEN_TYPES}'
        if cls_token_type == 'pos':
            assert isinstance(cls_token_pos, (float, int))
            assert 1 >= cls_token_pos >= 0
        else:
            cls_token_pos = None
        
        token_count = int(token_grid_size ** 2) + cls_token_count
        
        hm = H_Matrix(
            t=token_grid_size,
            levels=mask_levels,
            mask_type=mask_type,
            cls_token_count=cls_token_count,
            cls_token_pos=cls_token_pos,
        )
        indexed_mask = hm.indexed_mask
        _shape = list(indexed_mask.shape)
        assert (len(_shape) == 2 and _shape[0] == _shape[1] == token_count
            ), f'indexed_mask.shape={_shape} != token_count[{token_count}]'
        
        masks, mask_base, mask_levels_final = cls.process_masks(
            indexed_mask=indexed_mask,
            mask_levels=mask_levels,
            cls_token_type=cls_token_type,
            cls_token_count=cls_token_count,
        )
        
        print(f'type[{mask_type}] global[{"?"}] cls[{cls_token_type}] mask_levels_final[{mask_levels_final}]')
        print(f'masks{list(masks.shape)} mask_base{list(mask_base.shape)}')
        
        return masks, mask_base, mask_levels_final
    
    
    @classmethod
    def get_masks_1d(cls,
                token_count=16,
                mask_type='h',
                cls_token_type='pos',
                cls_token_pos=0.5,
                cls_token_count=0,
                mask_levels=3,
                ):
        
        # assert mask_type in MASK_TYPES, f'mask_type[{mask_type}] must be one of {MASK_TYPES}'
        assert cls_token_type in CLS_TOKEN_TYPES, f'cls_token_type[{cls_token_type}] must be one of {CLS_TOKEN_TYPES}'
        if cls_token_type == 'pos':
            assert isinstance(cls_token_pos, (float, int))
            assert 1 >= cls_token_pos >= 0
        else:
            cls_token_pos = None
        
        hm = H_Matrix_1D(
            n=token_count,
            levels=mask_levels,
            mask_type=mask_type,
            cls_token_count=cls_token_count,
            cls_token_pos=cls_token_pos,
        )
        indexed_mask = hm.indexed_mask
        _shape = list(indexed_mask.shape)
        assert (len(_shape) == 2 and _shape[0] == _shape[1] == token_count
            ), f'indexed_mask.shape={_shape} != token_count[{token_count}]'
        
        masks, mask_base, mask_levels_final = cls.process_masks(
            indexed_mask=indexed_mask,
            mask_levels=mask_levels,
            cls_token_type=cls_token_type,
            cls_token_count=cls_token_count,
        )
        
        print(f'type[{mask_type}] global[{"?"}] cls[{cls_token_type}] mask_levels_final[{mask_levels_final}]')
        print(f'masks{list(masks.shape)} mask_base{list(mask_base.shape)}')
        
        return masks, mask_base, mask_levels_final

    
# %%
if __name__ == '__main__':
    
    # h (1d)
    masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks_1d(
        token_count=32,
        mask_type='h',
        # cls_token_type='pos',
        # cls_token_pos=0.5,
        cls_token_count=0,
        mask_levels=3,
    )
    masks.shape
    
    # cross
    a = (masks[0, 0] * np.arange(mask_levels_final)).sum(-1)
    Matrix.plot_matrix(a).show()
    
    masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks(
        token_grid_size=12,
        mask_type='cross',
        cls_token_type='pos',
        cls_token_pos=0.5,
        cls_token_count=0,
        mask_levels=3,
    )
    masks.shape
    
    a = (masks[0, 0] * np.arange(mask_levels_final)).sum(-1)
    Matrix.plot_matrix_grid(a).show()
    
    # distq with 1 CLS
    _cls_token_count = 1
    masks, mask_base, mask_levels_final = H_Matrix_Masks.get_masks(
        token_grid_size=12,
        mask_type='distq',
        cls_token_type='pos',
        cls_token_pos=0.5,
        cls_token_count=_cls_token_count,
        mask_levels=3,
    )
    masks.shape
    a = (masks[0, 0] * np.arange(mask_levels_final)).sum(-1)
    Matrix.plot_matrix(a).show()
    Matrix.plot_matrix_grid(a[_cls_token_count:, _cls_token_count:]).show()


# %%
