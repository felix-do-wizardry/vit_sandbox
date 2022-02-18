# %%
import numpy as np
import json

# %%
class H_Matrix:
    def __init__(self, t=14, level=2):
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
        
        assert isinstance(level, int)
        assert level > 0
        assert level % 2 == 0
        self.level = level
        w = t / 2 ** (level / 2)
        assert w % 1 == 0
        self.w = int(w)
        self.cw = int(w * w)
        # w: window/cell size | cw: number of tokens per window/cell
        # print(f'{w=}, {cw=}')
        
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
        
        match = np.zeros([n, n], dtype=int)
        match_h = np.zeros([n, n], dtype=int)
        dist = np.zeros([n, n], dtype=float)
        data = []
        for iq in range(n):
            pq = (int(iq // t), iq % t)
            for ik in range(n):
                pk = (int(ik // t), ik % t)
                _dist = np.sqrt(np.sum((np.array(pq) - np.array(pk)) ** 2)) / t
                
                x_match = int(cxf[iq] == cxf[ik])
                y_match = int(cyf[iq] == cyf[ik])
                
                _match = x_match + y_match
                _match_h = x_match * y_match + y_match
                
                assert _match in [0, 1, 2]
                match[iq, ik] = _match
                match_h[iq, ik] = _match_h
                
                dist[iq, ik] = _dist
                
                # data.append({
                #     'iq': iq,
                #     'ik': ik,
                #     'match': _match,
                #     'pq': pq,
                #     'pk': pk,
                # })
        
        # df = pd.DataFrame(data)
        # df
        dist_neg = -dist
        digitize_levels = 10
        dist_dig = np.clip(np.digitize(
            dist_neg,
            np.percentile(
                dist_neg,
                np.linspace(0, 100, digitize_levels + 1),
            ),
        ), 1, digitize_levels) - 1
        
        self.match = match
        self.match_h = match_h
        self.dist = dist
        self.dist_neg = dist_neg
        self.dist_dig = dist_dig

# %%
# hm = H_Matrix(
#     t=int(224 // 16),
#     level=2,
# )
# hm.match
# hm.match_h

# %%
