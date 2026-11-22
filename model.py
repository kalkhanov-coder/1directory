import numpy as np
from scipy.optimize import linprog
S=np.array([120,150,90,200,110,160,130,100,180,140.])
C=np.array([[4.8,8.0,6.2,11.5],[5.0,7.6,6.0,12.0],[4.6,8.4,5.8,10.8],[5.2,7.8,6.6,12.4],[4.7,8.2,5.6,11.0],
[5.1,7.4,6.4,11.8],[4.9,8.6,6.1,12.2],[4.5,7.9,5.9,10.5],[5.3,8.1,6.8,12.6],[4.8,7.7,6.3,11.2]])
W=np.array([[6.0,4.0,7.5,5.0],[6.2,4.2,7.2,5.2],[5.8,3.8,7.6,4.8],[6.4,4.4,7.0,5.4],[6.1,4.0,7.8,5.1],
[6.3,4.1,7.1,5.3],[5.9,3.9,7.4,5.0],[6.0,4.3,7.7,4.9],[6.5,4.5,6.9,5.5],[6.2,4.0,7.3,5.2]])
a=np.array([0.40,0,0.05,0]); b=np.array([0.60,0.25,0.30,0.20])
L=np.outer(S,a); H=np.outer(S,b); K,n=C.shape
def solve(F,C=C,W=W):
    c=-C.ravel(); A=[W.ravel()]; bb=[F]
    for k in range(K):
        r=np.zeros(K*n); r[k*n:(k+1)*n]=1; A.append(r); bb.append(S[k])
    res=linprog(c,A_ub=np.array(A),b_ub=bb,bounds=list(zip(L.ravel(),H.ravel())),method='highs')
    return -res.fun, res.x.reshape(K,n), -res.ineqlin.marginals[0], -res.ineqlin.marginals[1:]
def resp(u,eps=0.0,C=C,W=W):
    # greedy response, eps tie-break: evaluate at u+eps
    X=L.copy()
    for k in range(K):
        d=C[k]-(u+eps)*W[k]; free=S[k]-L[k].sum()
        for j in np.argsort(-d,kind='stable'):
            if d[j]<=0 or free<=0: break
            inc=min(H[k,j]-L[k,j],free); X[k,j]+=inc; free-=inc
    return X
D=lambda u,eps=0.0,C=C,W=W:(W*resp(u,eps,C,W)).sum()
