# %%
import numpy as np
import json

# %%
class H_Matrix:
    def __init__(self, t=14, level=2, mask_type='h'):
        '''Args:
        t (int): number of token columns/rows, assuming square input
        level (int): level of h matrix
        '''
        assert isinstance(t, int)
        assert t > 0
        self.t = t
        n = t * t
        self.n = n
        # print(f'{s=} {p=} {t=} {n=}')
        
        self.is_dist = mask_type in ['dist']
        self.mask_type = mask_type
        
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
        c.shape
        
        cy = (np.floor(y / w)).astype(int)
        cx = (np.floor(x / w)).astype(int)
        
        cy, cx
        
        cxf = cx.reshape(-1)
        cyf = cy.reshape(-1)
        cxf == cyf
        
        match   = np.zeros([n + 1, n + 1], dtype=int) - 1
        match_h = np.zeros([n + 1, n + 1], dtype=int) - 1
        dist    = np.zeros([n + 1, n + 1], dtype=float)
        data = []
        for iq in range(n):
            pq = (int(iq // t), iq % t)
            for ik in range(n):
                pk = (int(ik // t), ik % t)
                
                if self.is_dist:
                    _dist = np.sqrt(np.sum((np.array(pq) - np.array(pk)) ** 2)) / t
                    dist[iq + 1, ik + 1] = _dist
                else:
                    x_match = int(cxf[iq] == cxf[ik])
                    y_match = int(cyf[iq] == cyf[ik])
                    
                    _match = x_match + y_match
                    _match_h = x_match * y_match + y_match
                    
                    assert _match in [0, 1, 2]
                    
                    match[iq + 1, ik + 1] = _match
                    match_h[iq + 1, ik + 1] = _match_h
                
        
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
        

# %%
if __name__ == '__main__':
    hm = H_Matrix(
        t=int(224 // 16),
        level=2,
    )
    hm.match
    hm.match_h

# %%
