import sys,time,json;sys.path.insert(0,'.')
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from exact import exact,response
def inst(K,n,rng):
    S=rng.uniform(80,220,K)
    W=rng.uniform(3.5,8.0,(K,n)); C=W*rng.uniform(0.6,2.4,(K,n))
    a=np.zeros(n); a[0]=0.4; a[min(2,n-1)]=0.05
    b=np.full(n,min(0.6,1.5/n+0.1)); b[0]=0.6
    L=np.outer(S,a); H=np.maximum(np.outer(S,b),L)
    return C,W,L,H,S
def highs(C,W,L,H,S,F):
    K,n=C.shape; N=K*n
    rows=[np.zeros(N,int)]; cols=[np.arange(N)]; vals=[W.ravel()]
    rows.append(1+np.repeat(np.arange(K),n)); cols.append(np.arange(N)); vals.append(np.ones(N))
    A=sp.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(K+1,N))
    r=linprog(-C.ravel(),A_ub=A,b_ub=np.r_[F,S],bounds=np.c_[L.ravel(),H.ravel()],method='highs')
    return -r.fun,-r.ineqlin.marginals[0]
if __name__=='__main__':
  rng=np.random.default_rng(2026); out=[]
  for n in [4,10]:
      for K in [10,100,1000,10000,100000]:
          C,W,L,H,S=inst(K,n,rng); F=0.9*(W*response(1e-12,C,W,L,H,S)).sum()
          te=[];th=[]
          for rep in range(3):
              t=time.perf_counter(); u,X=exact(C,W,L,H,S,F); te.append(time.perf_counter()-t)
              t=time.perf_counter(); z,uh=highs(C,W,L,H,S,F); th.append(time.perf_counter()-t)
          ze=(C*X).sum(); res=dict(n=n,K=K,vars=K*n,t_exact=float(np.median(te)),t_highs=float(np.median(th)),relgap=abs(ze-z)/abs(z),du=abs(u-uh),viol=(W*X).sum()-F)
          out.append(res); print(res,flush=True)
  json.dump(out,open('bench.json','w'),indent=1)
