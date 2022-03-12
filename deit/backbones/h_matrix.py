# %%
import numpy as np
import json

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
        
        assert mask_type in ['dist', 'hdist', 'dist']
        self.mask_type = mask_type
        self.is_dist = mask_type in ['dist']
        
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
        data = []
        for iq in range(n_img):
            pq = (int(iq // t), iq % t)
            
            for ik in range(n_img):
                pk = (int(ik // t), ik % t)
                
                if self.is_dist:
                    _dist = np.sqrt(np.sum((np.array(pq) - np.array(pk)) ** 2)) / t
                    dist[iq, ik] = _dist
                else:
                    x_match = int(cxf[iq] == cxf[ik])
                    y_match = int(cyf[iq] == cyf[ik])
                    
                    _match = x_match + y_match
                    _match_h = x_match * y_match + y_match
                    
                    assert _match in [0, 1, 2]
                    
                    match[iq, ik] = _match
                    match_h[iq, ik] = _match_h
        
        assert np.all(match >= 0), f'?? {np.min(match)}'
        assert np.all(match_h >= 0), f'?? {np.min(match_h)}'
        
        match = np.pad(match, pad_width=self.cls_token_pad, constant_values=0)
        
        dist_neg = -dist
        digitize_levels = level + 1
        dist_dig = np.clip(np.digitize(
            dist_neg,
            np.percentile(
                dist_neg,
                np.linspace(0, 100, digitize_levels + 1),
            ),
        ), 1, digitize_levels) - 1
        dist_dig[0, :] = -1
        dist_dig[:, 0] = -1
        
        # shape = [t*t+1, t*t+1], range = [-1, level]
        self.match = match
        self.match_h = match_h
        self.dist = dist
        self.dist_neg = dist_neg
        self.dist_dig = dist_dig
        
        
        if mask_type in ['h']:
            self.indexed_mask = self.match_h
        elif mask_type in ['hdist']:
            self.indexed_mask = self.match
        elif mask_type in ['dist']:
            self.indexed_mask = self.dist_dig
        else:
            raise NotImplementedError(f'`mask_type`[{mask_type}] has not been implemented')
        
    # @classmethod
    # def check_cls(cls, n=4, i=0, count=1, order='first'):
    #     if count < 1:
    #         return False
    #     if order == 'first':
    #         return i < count

# # %%
# if __name__ == '__main__':
#     hm = H_Matrix(
#         t=int(224 // 16),
#         level=2,
#     )
#     print(hm.match)
#     hm.match_h

# # %%
# assert 0, 'DEBUGING'

# %%
_token_grid_size = 14
_mask_levels = 3
_mask_type = 'dist'

hm = H_Matrix(
    t=_token_grid_size,
    level=_mask_levels - 1,
    mask_type=_mask_type,
    
    with_cls_token=True,
    cls_token_pos=0.5,
)
indexed_mask = hm.indexed_mask
indexed_mask

# %%
indexed_mask.shape, np.percentile(indexed_mask, np.linspace(0, 100, 11))

# %%
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# %%
fig = px.imshow(
    indexed_mask,
)
fig.update_layout(
    template='plotly_dark',
    
)
fig

# %%
np.pad()