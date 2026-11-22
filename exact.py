"""Exact equilibrium price (Theorem 3.3), vectorized implementation."""
import numpy as np
def response(u,C,W,L,H,S):
    d=C-u*W; K,n=C.shape
    order=np.argsort(-d,axis=1,kind='stable')
    ds=np.take_along_axis(d,order,1); cap=np.take_along_axis(H-L,order,1)
    cap=np.where(ds>0,cap,0.0)
    free=(S-L.sum(1))[:,None]
    cum=np.cumsum(cap,1); prev=cum-cap
    inc=np.clip(np.minimum(cap,free-prev),0,None)
    X=L.copy(); np.put_along_axis(X,order,np.take_along_axis(X,order,1)+inc,1)
    return X
def breakpoints(C,W):
    K,n=C.shape; vals=[C/W]
    for j in range(n):
        for l in range(j+1,n):
            dw=W[:,j]-W[:,l]; ok=dw!=0
            v=np.full(K,np.nan); v[ok]=(C[ok,j]-C[ok,l])/dw[ok]; vals.append(v[:,None])
    B=np.concatenate([v.ravel() for v in vals]); B=B[np.isfinite(B)&(B>0)]
    return np.unique(B)
def exact(C,W,L,H,S,F,eps=1e-10):
    D=lambda u:(W*response(u,C,W,L,H,S)).sum()
    if D(eps)<=F:
        X=response(eps,C,W,L,H,S); return 0.0,X
    B=breakpoints(C,W); lo,hi=0,len(B)-1          # find min b with D(b+)<=F by bisection
    while lo<hi:
        m=(lo+hi)//2
        if D(B[m]*(1+eps)+eps)<=F: hi=m
        else: lo=m+1
    b=B[lo]; Xm=response(b*(1-eps)-eps,C,W,L,H,S); Xp=response(b*(1+eps)+eps,C,W,L,H,S)
    Dm,Dp=(W*Xm).sum(),(W*Xp).sum(); th=(F-Dp)/(Dm-Dp) if Dm>Dp else 0.0
    return b,th*Xm+(1-th)*Xp
