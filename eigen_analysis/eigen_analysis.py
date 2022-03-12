import numpy as np
# import tqdm

pre='softmax'

def percent(X,per):
    total=np.sum(X)
    for i in range(1,X.shape[0]+1):
        if(np.sum(X[-i:])/total>=per):
            return i

num_comps=[]
# for i in tqdm.tqdm(range(16)):
for i in range(16):
    V = np.load('/tanData/cov_matrices/mixtureqk2soft_100k_V'+str(i)+'.npy')
    W = np.load('/tanData/cov_matrices/mixtureqk2soft_100k_W'+str(i)+'.npy')
    #print(np.sum(W0[-10:])/np.sum(W0))
    num_comps.append(percent(W,0.95))
print(num_comps)
