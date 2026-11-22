import sys;sys.path.insert(0,'.')
from stoch import *
rm=lambda x:np.sqrt(((x-uE)**2).mean())
res={}
for a in [1e-4,2e-4,4e-4,8e-4,1.6e-3,3.2e-3,6.4e-3]:
    U=run(lambda s:a/s); res[('a',a)]=(rm(U[:,-1]),rm(U[:,1000:].mean(1)))
    print('alpha/s',a,*np.round(res[('a',a)],4))
for th in [3e-6,1e-5,3e-5,1e-4,3e-4,1e-3]:
    U=run(lambda s:th/np.sqrt(s)); res[('t',th)]=(rm(U[:,-1]),rm(U[:,1000:].mean(1)))
    print('theta/sqrt',th,*np.round(res[('t',th)],4))
import pickle;pickle.dump(res,open('sweep.pkl','wb'))
