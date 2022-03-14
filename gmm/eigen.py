# %%
import numpy as np
import json
import time

# from scipy.linalg import eigh
import torch
eigh = torch.linalg.eigh
eigvals = torch.linalg.eigvals

# %%
# A = cov_matrices[i]
# W, V = eigh(A)
# np.save('/tanData/cov_matrices/'+name+'_100k_W'+str(i),W)
# np.save('/tanData/cov_matrices/'+name+'_100k_V'+str(i),V)

# %%
# t = 8
# n = t * t + 1
# a = np.random.sample([n, n])
# a.shape

# %%
def calc_eigh(attn):
    time_start = time.time()
    
    # a = torch.tensor(attn, device='cuda')
    a = attn
    _b = a.reshape(-1, 1) * a.reshape(1, -1)
    
    time_mult = time.time() - time_start
    
    time_start = time.time()
    b = torch.tensor(_b, device='cuda')
    W, V = eigh(b)
    W, V = eigh(b)
    
    time_eigh = time.time() - time_start
    time_total = time_mult + time_eigh
    
    del W
    del V
    del b
    del _b
    return time_mult, time_eigh, time_total
    # return W, V, time_mult, time_eigh, time_total

# %%
# for n in [145]:
#     t = round(np.sqrt(n-1), 2)
# for i, t in enumerate(range(10, 13)):
for i, t in enumerate(range(4, 9)):
    if i > 0: time.sleep(4)
    
    n = t * t + 1
    attn = np.random.sample([n, n])
    
    # W, V, time_mult, time_eigh, time_total = calc_eigh(attn)
    
    torch.cuda.empty_cache()
    time_start = time.time()
    _ = calc_eigh(attn)
    time_total = time.time() - time_start
    
    # mem = torch.cuda.max_memory_allocated()
    # print(f't[{t}] n[{n}] attn[{attn.shape}] b[{b.shape}] W[{W.shape}] V[{V.shape}]')
    # print(f'time: total[{time_total:.2f}s] mult[{time_mult:.2f}s] eigh[{time_eigh:.2f}s]')
    print()
    print(f't[{t}] n[{n}] attn[{list(attn.shape)}] time_total[{time_total:.2f}s]')
    del attn

# %%
import sys
sys.exit()

# %%
import pandas as pd
import plotly.express as px

# %%
df = pd.read_csv('/home/hai/vit_sandbox/gmm/eigen_log.csv')
df

# %%
fig = px.line(
    df,
    x='t',
    # y=['time', 'mem'],
    y='time',
    template='plotly_dark',
    height=320,
    log_y=True,
)
fig.update_layout(margin=dict(t=0,b=0,l=0,r=0)).show()
fig = px.line(
    df,
    x='t',
    y='mem',
    template='plotly_dark',
    height=320,
    log_y=True,
)
fig.update_layout(margin=dict(t=0,b=0,l=0,r=0)).show()
