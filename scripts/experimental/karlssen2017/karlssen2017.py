
#%%
import numpy as np
from dmpbbo.dmp.Dmp import *
from dmpbbo.dynamicalsystems.ExponentialSystem import ExponentialSystem



class FunctionApproximatorKarlssen:
    def __init__(self, n_kernel, alpha_x):
        
        self.n_kernel = n_kernel
        
        ct = np.atleast_2d(np.linspace(0, 1, n_kernel)).T
        cx = np.exp(-alpha_x*3*ct)
        c = cx
        self.c = cx
        
        d = np.power(np.diff(c,axis=0)*0.55,2)
        d = 1/np.append(d, d[-1])
        self.d = np.atleast_2d(d).T
        
    
    def train(self,inputs,targets, s):
        self.inputs = inputs
        self.targets = targets
        self.s = s
        p = inputs.shape[0]
        c = self.c
        d = self.d
        n_kernel = self.n_kernel
        
        psi = np.zeros((p,n_kernel))
        
        inputs = np.squeeze(inputs)
        
        for t in range(0, p):
            psi[t,:] = np.squeeze(np.exp(-0.5*np.power(((inputs[t]-c)),2) * d))
            
        w = np.zeros((n_kernel,1))
        
        for i in range(0, n_kernel):
            gamma_i = np.diag(psi[:,i])
            w[i] = np.transpose(s) @ gamma_i @ targets @ np.linalg.inv(np.transpose(s) @ gamma_i @ s)
            
        self.w = w
        
        
    def predict(self, input):
        c = self.c
        d = self.d
        w = self.w
        
        psi = np.exp(-0.5*np.power(((input-c)),2) * d)
        f = np.sum(np.transpose(w) @ psi) / (np.sum(psi+ 10 ** -10))
        return f


class MyDmp:
    def __init__(self, y0, g, tau, alpha_z, beta_z, alpha_x, n_kernel, function_approximators=None):
        
        self.dim = y0.size
        self.y0 = y0
        self.g = g
        self.tau = tau
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.alpha_x = alpha_x
        
        
        self.function_approximators = []
        
        for i in range(0, self.dim):
            self.function_approximators.append(FunctionApproximatorKarlssen(n_kernel, alpha_x))
        
    
    def train(self,trajectory):
        
        self.p = trajectory.ts_.size       
        
        
        self.f_target = np.power(self.tau,2)*trajectory.ydds_ - self.alpha_z*(self.beta_z*(self.g-trajectory.ys_) -self.tau*trajectory.yds_)
        self.phase_system = ExponentialSystem(self.tau, 1,0,1)
        
        ### X integration matlab code style
        
        # self.xs = np.zeros((self.p,1))
        # self.xs[0] = 1
        
        # for t in range(1, self.p):
        #     dt = 1/250
        #     xdot = -self.alpha_x*self.xs[t-1]/self.tau
        #     self.xs[t] = self.xs[t-1] + xdot*dt
        
        # ## X integration, a more modern style
        # self.xs = np.zeros((self.p,1))
        # self.xds = np.zeros((self.p,1))
        
        # (self.xs[0],self.xds[0]) = self.phase_system.integrateStart()
        
        # for t in range(1, self.p):
        #     dt = 1/250
        #     (self.xs[t],self.xds[t]) = self.phase_system.integrateStep(dt,self.xs[t-1])
        
        ## X integration, Analytical solution
        (self.xs,self.xds) = self.phase_system.analyticalSolution(trajectory.ts_)
        
        
        self.s = self.xs*(self.g-self.y0)
        
        
        for dd in range(0, self.dim):
            targets = self.f_target[:,dd][:,np.newaxis]
            ss = self.s[:,dd][:,np.newaxis]
            self.function_approximators[dd].train(self.xs,targets,ss)

    def integrateStart(self):
        (self.x,self.xd) = self.phase_system.integrateStart()
        self.ydot = np.zeros(self.dim)
        self.y = self.y0
        self.yddot = np.zeros(self.dim)
        self.t = 0
        
        return (self.x,self.y,self.ydot,self.yddot)
        
    def integrateStep(self, t):
        self.dt = t - self.t
        self.t = t
        
        self.dmp2acc()
        self.ydot = self.ydot + self.yddot*self.dt
        self.y = self.y + self.ydot*self.dt
        (self.x,self.xd) = self.phase_system.integrateStep(self.dt,self.x)
        
        return (self.x,self.y,self.ydot,self.yddot)
        
    def dmp2acc(self):
        self.f = np.zeros(self.dim)
        for dd in range(0, self.dim):
            self.f[dd] = self.function_approximators[dd].predict(self.x)
            self.f[dd] = self.f[dd] * self.x * (self.g-self.y0)[dd]
        
        self.yddot = 1/(self.tau ** 2) * (self.alpha_z*(self.beta_z*(self.g-self.y) - self.tau*self.ydot) + self.f)
        
#%%

# myclass = FunctionApproximatorKarlssen(15, 1)
# myclass.train(x,f_target,s)


#%%

alpha_e = 5
alpha_z = 25
alpha_x = 1
beta_z = 6.25

n_kernel = 15

tau = trajectory.ts_[-1]/3
y0 = trajectory.ys_[0,:]
g = trajectory.ys_[-1,:]
dmp = MyDmp(y0, g, tau, alpha_z, beta_z, alpha_x, n_kernel)
dmp.train(trajectory)


#%%

time = trajectory.ts_[-1]
dt = 1/250
n_steps = int(time/dt)
ts = np.linspace(0,time,n_steps)

xs = np.zeros(ts.size)
ys = np.zeros((ts.size, dmp.dim))
ydots = np.zeros((ts.size, dmp.dim))
yddots = np.zeros((ts.size, dmp.dim))

(xs[0], ys[0], ydots[0], yddots[0]) = dmp.integrateStart()
t = 0

#%%

for i in range(1, ts.size):
    (xs[i], ys[i], ydots[i], yddots[i]) = dmp.integrateStep(ts[i])

#%%


fig = plt.figure(1)
axs = [ fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
for i in range(0,len(axs)):
    axs[i].plot(trajectory.ts_, trajectory.ys_[:,i])
    axs[i].plot(ts, ys[:,i])
    

#%%
plt.plot(trajectory.ts_, trajectory.ys_[:,0])
plt.plot(ts, ys[:,0])

