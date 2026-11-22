import sys;sys.path.insert(0,'.')
from model import *
p=np.array([0.3,0.5,0.2]); mc=np.array([0.90,1.0,1.08]); mw=np.array([1.15,1.0,0.90])
cb=(p*mc).sum(); wb=(p*mw).sum()
Cb,Wb=C*cb,W*wb
F=7000
ZE,XE,uE,_=solve(F,Cb,Wb)
# breakpoints of response under (Cb,Wb)
bp=set()
for k in range(K):
  for j in range(n):
    bp.add(Cb[k,j]/Wb[k,j])
    for l in range(j+1,n):
      if Wb[k,j]!=Wb[k,l]:
        v=(Cb[k,j]-Cb[k,l])/(Wb[k,j]-Wb[k,l])
        if v>0: bp.add(v)
bp=np.array(sorted(bp)); mids=np.r_[bp[0]/2,(bp[:-1]+bp[1:])/2,bp[-1]+1]
WX=np.array([(W*resp(m,0,Cb,Wb)).sum() for m in mids])  # sum w_kj x_kj with nominal w
def Dobs(u,om): return mw[om]*WX[np.searchsorted(bp,u)]
def run(rule,T=2000,R=150,seed=1):
    rng=np.random.default_rng(seed); U=np.zeros((R,T))
    for r in range(R):
        u=0.0; oms=rng.choice(3,size=T,p=p)
        for s in range(T):
            U[r,s]=u; u=max(0.0,u+rule(s+1)*(Dobs(u,oms[s])-F))
    return U
if __name__=='__main__':
    print('cb wb',cb,wb,'uE',uE,'ZE',ZE)
    rm=lambda x:np.sqrt(((x-uE)**2).mean())
    U=run(lambda s:8e-4/s); print('1/s last mean',U[:,-1].mean(),'rmse',rm(U[:,-1]))
    U=run(lambda s:3e-5/np.sqrt(s)); last=U[:,-1]; avg=U[:,1000:].mean(1)
    print('1/sqrt last rmse',rm(last),'avg rmse',rm(avg),'avg mean',avg.mean())
