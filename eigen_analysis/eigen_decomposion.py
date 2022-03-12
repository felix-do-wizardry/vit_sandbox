from scipy.linalg import eigh
import numpy as np 
import tqdm
for name in ['mixtureqk2soft']:#['qk4','qk4relu','qk4relusoft','qk2','qk2relu','qk2relusoft']:#,'softmax']:
    cov_matrices=np.load('/tanData/cov_matrices/'+name+'_100k_variances.npy')
    for i in tqdm.tqdm(range(16)):
        A = cov_matrices[i]
        W, V = eigh(A)
        np.save('/tanData/cov_matrices/'+name+'_100k_W'+str(i),W)
        np.save('/tanData/cov_matrices/'+name+'_100k_V'+str(i),V)
    A = cov_matrices[0]
    for i in range(1,16):
        A+=cov_matrices[i]
    W, V = eigh(A)
    np.save('/tanData/cov_matrices/'+name+'_100k_Wall',W)
    np.save('/tanData/cov_matrices/'+name+'_100k_Vall',V)

