import sys;sys.path.insert(0,'.')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
import dec, stoch
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':10,'axes.linewidth':0.8})
F=7000; us=dec.us; Zs=dec.Zs
# Fig1 demand
uu=np.linspace(0,2.5,5001); DD=[D(u,1e-12) for u in uu]
fig,ax=plt.subplots(figsize=(5.2,3.1))
ax.step(uu,DD,where='post',color='k',lw=1.2,label=r'$D(u)$')
ax.axhline(F,ls='--',color='k',lw=0.8,label=r'$F=7000$')
ax.axvline(us,ls=':',color='k',lw=0.8); ax.text(us+0.02,7550,r'$u^*$')
ax.set_xlabel(r'water price $u$ (million UZS per thousand m$^3$)'); ax.set_ylabel(r'demand (thousand m$^3$)')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig('fig/fig1_demand.pdf')
# Fig2
U,Xl=dec.run(300,lambda s:8e-4/s)
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.4,2.6))
s=np.arange(1,301); a1.semilogx(s,U,'k',lw=1); a1.axhline(us,ls='--',color='k',lw=0.8); a1.set_ylim(0.6,1.0)
a1.set_xlabel('iteration $s$'); a1.set_ylabel('$u^s$'); a1.set_title('(a) price iterates',fontsize=10)
t=np.arange(2,301,2); viol=[];gap=[]
for tt in t:
    xh=Xl[tt//2:tt].mean(0); viol.append(abs((W*xh).sum()-F)); gap.append(abs(Zs-(C*xh).sum()))
a2.loglog(t,viol,'k',lw=1,label=r'$|\sum w\hat x^t-F|$'); a2.loglog(t,gap,'k--',lw=1,label=r'$|Z^*-\sum c\hat x^t|$')
a2.set_xlabel('iteration $t$'); a2.set_title('(b) trailing-window averages',fontsize=10); a2.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig/fig2_decentralized.pdf')
# Fig3
rm=lambda A:np.sqrt(((A-stoch.uE)**2).mean(0))
U1=stoch.run(lambda s:8e-4/s); U2=stoch.run(lambda s:3e-5/np.sqrt(s))
cs=np.cumsum(U2,1); T=U2.shape[1]; idx=np.arange(1,T+1); h=idx//2
avg=np.array([(cs[:,i]-(cs[:,h[i]-1] if h[i]>0 else 0))/(i+1-h[i]) for i in range(T)]).T
fig,ax=plt.subplots(figsize=(5.2,3.1))
ax.loglog(idx,rm(U1),'k:',lw=1.2,label=r'$\rho_s=8\cdot10^{-4}/s$ (last iterate)')
ax.loglog(idx,rm(U2),'k--',lw=1,label=r'$\rho_s=3\cdot10^{-5}/\sqrt{s}$ (last iterate)')
ax.loglog(idx,rm(avg),'k',lw=1.2,label=r'$\rho_s=3\cdot10^{-5}/\sqrt{s}$, trailing average')
ax.set_xlabel('iteration $s$'); ax.set_ylabel('RMSE of price'); ax.legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig/fig3_stochastic.pdf')
# Fig4
Fs=np.linspace(4000,7900,400); Zv=[];uv=[]
for f in Fs:
    z,_,u,_=solve(f); Zv.append(z); uv.append(u)
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.4,2.6))
a1.plot(Fs,Zv,'k',lw=1); a1.set_xlabel(r'$F$ (thousand m$^3$)'); a1.set_ylabel(r'$Z^*(F)$ (million UZS)'); a1.set_title('(a) value function',fontsize=10)
a2.step(Fs,uv,'k',lw=1,where='mid'); a2.axvline(6782.65,ls=':',color='k',lw=0.8); a2.axvline(F,ls='--',color='k',lw=0.8)
a2.set_xlabel(r'$F$ (thousand m$^3$)'); a2.set_ylabel(r'$u^*(F)$'); a2.set_title('(b) equilibrium price',fontsize=10)
fig.tight_layout(); fig.savefig('fig/fig4_parametric.pdf')
print('ok', rm(U1)[-1], rm(U2)[-1], rm(avg)[-1], avg[:,-1].mean(), U1[:,-1].mean())
