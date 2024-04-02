import numpy as np



n= 10
M = 20*(np.random.rand(n,n)-0.5)
b = 20*(np.random.rand(n)-0.5)

U,S,VT = np.linalg.svd(M)

# print(S/S[0])
for eps in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
    k = np.searchsorted(S[::-1], S[0]*eps, side='right')
    I = (S>S[0]*(1-eps))
    print(f"{I=} {k=}")
    Mk = (U[:,:k]*S[:k])@VT[:k,:]
    M2 = (U[:,I]*S[I])@VT[I,:]
    print(f"{eps=} {k=} {np.linalg.norm(M-Mk)=} {np.linalg.norm(M-M2)=}")
