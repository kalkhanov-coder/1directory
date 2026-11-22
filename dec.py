import sys;sys.path.insert(0,'.')
from model import *
F=7000; Zs,Xs,us,_=solve(F)
def run(T,rule,u1=0.0):
    u=u1; U=[];Xl=[]
    for s in range(1,T+1):
        X=resp(u); U.append(u); Xl.append(X)
        u=max(0,u+rule(s)*((W*X).sum()-F))
    return np.array(U),np.array(Xl)
if __name__=='__main__':
    U,Xl=run(300,lambda s:8e-4/s)
    print('u3',U[2],'last bad', max([s+1 for s in range(300) if abs(U[s]-us)>2.5e-4]))
    t=300; xh=Xl[t//2-1:t].mean(0)
    print('window viol',(W*xh).sum()-F,'gap',Zs-(C*xh).sum(),'x43',xh[3,2], 'len',t-t//2+1)
    rho=np.array([8e-4/s for s in range(1,t+1)]); xw=(rho[:,None,None]*Xl).sum(0)/rho.sum()
    print('weighted viol',(W*xw).sum()-F)
