
#%%
from lfd_dmp.karlsson2017 import DMPkarlsson


#%%

alpha_e = 5
alpha_z = 25
alpha_x = 1
beta_z = 6.25
kc = 10000

n_kernel = 15

tau = trajectory.ts_[-1]
y0 = trajectory.ys_[0,:]
g = trajectory.ys_[-1,:]
dmp = DMPkarlsson(y0, g, tau, alpha_z, beta_z, alpha_x, n_kernel,kc,alpha_e)
dmp.train(trajectory)


#%% Change initial and attractor state
dmp.set_initial_state(dmp.y0 + 0.1)
dmp.set_attractor_state(dmp.g + 0.2)

#%%

time = 2 * trajectory.ts_[-1]
dt = 1/250
n_steps = int(time/dt)
ts = np.linspace(0,time-dt,n_steps)

xs = np.zeros(ts.size)
ys = np.zeros((ts.size, dmp.dim))
ydots = np.zeros((ts.size, dmp.dim))
yddots = np.zeros((ts.size, dmp.dim))

(xs[0], ys[0], ydots[0], yddots[0]) = dmp.integrateStart(tau)
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

y_unpert = ys



#%%

time = 2 * trajectory.ts_[-1]
dt = 1/250
n_steps = int(time/dt)
ts = np.linspace(0,time-dt,n_steps)

xs = np.zeros(ts.size)
ys = np.zeros((ts.size, dmp.dim))
yas = np.zeros((ts.size, dmp.dim))

(xs[0], ys[0], yas[0]) = dmp.controlStart(0.2* tau)
t = 0

#%%


ya = dmp.y0
ya_dot = np.zeros(dmp.dim)

        
# t = 0
for t in range(1, ts.size):
    # t = t+1
    #Create our fake "actual" trajectory and basically input ya and ya_dot to the main algorithm
    ya_dot = ya_dot + dmp.ydd_a*dt
    if (t>499 and t<3749):
        # ya_ddot_perturbation = -25*ya_dot
        ya_ddot_perturbation = (.30-ya_dot)*25
        ya_dot = ya_dot + ya_ddot_perturbation*dt
    
    ya = ya + ya_dot*dt
    
    (xs[t], ys[t], yas[t]) = dmp.controlStep(ts[t], ya, ya_dot)
    
    
#%%

plt.plot(ts, y_unpert[:,0])
plt.plot(ts, ys[:,0])
plt.plot(ts, yas[:,0])