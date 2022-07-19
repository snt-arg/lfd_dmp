#%%
# Import stuff
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


def traj2w(y_demo, dt, tau, c, d, alpha_z, beta_z, alpha_x, n_kernel):
    p = len(y_demo)
    g = y_demo[-1]
    y0 = y_demo[0]
    ydot_demo = np.append(np.diff(y_demo,axis=0),0)[:,np.newaxis]/dt
    yddot_demo = np.append(np.diff(ydot_demo, axis=0), 0)[:,np.newaxis]/dt

    f_target = np.power(tau,2)*yddot_demo -alpha_z*(beta_z*(g-y_demo) -tau*ydot_demo)
    
    x = np.zeros(y_demo.shape)
    x[0] = 1
    
    for t in range(1, p):
        x_dot = -alpha_x*x[t-1]/tau
        x[t] = x[t-1] + x_dot*dt
    
    psi = np.zeros((p,n_kernel))
    
    for t in range(0, p):
        psi[t,:] = np.squeeze(np.exp(-0.5*np.power(((x[t]-c)),2) * d))
    
    s = x*(g-y0)
    
    w = np.zeros((n_kernel,1))
    
    for i in range(0, n_kernel):
        gamma_i = np.diag(psi[:,i])
        w[i] = np.transpose(s) @ gamma_i @ f_target @ np.linalg.inv(np.transpose(s) @ gamma_i @ s)
        
    return w

def dmp2acc(y0 , y , ydot , g , tau , w , x , c , d , alpha_z , beta_z):
    
    psi = np.exp(-0.5*np.power(((x-c)),2) * d)
    f = np.sum(np.transpose(w) @ psi) / (np.sum(psi+ 10 ** -10)) * x * (g-y0)
    yddot = 1/(tau ** 2) * (alpha_z*(beta_z*(g-y) - tau*ydot) + f)
    
    return yddot

def get_ya_ddot_lowgain_ff(ya, ya_dot, y, ydot, yddot):
    kp = 25
    kv = 10
    ya_ddot = kp*(y-ya) + kv*(ydot-ya_dot) + yddot
    return ya_ddot


z = 0

def dmp2vel_acc_ss(y0, y, g, tau_adapt, w, x, dt, alpha_e, c, d, alpha_z, beta_z,ya,e,kc,tau):
    global z
    psi = np.exp(-0.5*np.power(((x-c)),2) * d) 
    f = np.sum(np.transpose(w) @ psi) / (np.sum(psi+ 10 ** -10)) * x * (g-y0)
    z_dot = 1/tau_adapt * (alpha_z*(beta_z*(g-y) - z) + f)
    z = z + dt*z_dot
    ydot = z/tau_adapt
    yddot = (z_dot*tau_adapt-tau*z*2*kc*e*(alpha_e*(ya-y-e)))/tau_adapt**2;
    
    return (ydot, yddot)

# Params
alpha_e = 5
alpha_z = 25
alpha_x = 1
beta_z = 6.25

n_kernel = 15

# arr = np.linspace(0,1,n_kernel)[:,np.newaxis] * 0.5
ct = np.linspace(0, 1, n_kernel)[:,np.newaxis]
cx = np.exp(-alpha_x*3*ct)
c = cx
kc = 10000
d = np.power(np.diff(c,axis=0)*0.55,2)
d = 1/np.append(d, d[-1])[:,np.newaxis]

## Demonstration Trajectory, 1 DOF
dt = 1/250
n_steps = int(10/dt)
t = np.linspace(0,10,n_steps)[:,np.newaxis]
traj = np.fmax(0, -np.sin(2*np.pi*t[0:1000]/2.5))
traj = np.concatenate((traj, traj[-1] * np.ones((50,1))))

# Determine DMP Demonstration
t_end = len(traj)*dt
tau = t_end/3
p = len(traj)
g = traj[-1]

# Determine the weight (traj2w)
w = traj2w(traj,dt,tau,c,d,alpha_z,beta_z,alpha_x,n_kernel)

#%%

# Unperturbed Trajectory
x = 1
ydot = 0
y = np.zeros((2*p,1))
y[0] = traj[0]
y0 = y[0]
yddot_unpert = np.zeros((2*p,1))

for t in range(1, 2*p):
    yddot = dmp2acc(y0, y[t-1], ydot, g, tau, w, x, c, d, alpha_z, beta_z)
    ydot = ydot + yddot*dt
    y[t] = y[t-1] + ydot*dt
    xdot = -alpha_x*x/tau
    x = x + xdot*dt
    yddot_unpert[t] = yddot

y_unpert = y;

#%%

# Perturbed Trajectory

x = 1
y = np.zeros((2*p,1))
y[0] = traj[0]
y0 = y[0]
ya = np.zeros((2*p,1))
ya_dot = 0
e = 0
ya_ddot_log = np.zeros(y.shape)
y_ddot_log = np.zeros(y.shape)
yddot = 0
ydot = 0
# tau_adapt = tau * (1+(kc*(e**2)))

ya_ddot = 0

#%%

# t = 0

for t in range(1, 2*p):
    # t = t + 1
    
    #Create our fake "actual" trajectory and basically input ya and ya_dot to the main algorithm
    ya_dot = ya_dot + ya_ddot*dt
    if (t>499 and t<749):
        ya_ddot_perturbation = -25*ya_dot
        ya_dot = ya_dot + ya_ddot_perturbation*dt
    
    ya[t] = ya[t-1] + ya_dot*dt
    ya_ddot_log[t] = ya_ddot
    y_ddot_log[t] = yddot
    
    
    
    tau_adapt = tau * (1+(kc*(e**2)))
    (ydot , yddot) = dmp2vel_acc_ss(y0, y[t-1], g, tau_adapt, w, x, dt, alpha_e, c, d, alpha_z, beta_z, ya[t-1], e, kc, tau)
    y[t] = y[t-1] + ydot*dt
    e_dot = alpha_e*(ya[t]-y[t]-e)
    e = e + e_dot*dt
    ya_ddot = get_ya_ddot_lowgain_ff(ya[t-1], ya_dot, y[t-1], ydot, yddot)
    
    xdot = -alpha_x*x/tau_adapt
    x = x + xdot*dt
    
#%%
t = np.cumsum(dt*np.ones(y.shape))

#%%
# Plot

fig = plt.figure(1)
axs = [ fig.add_subplot(211), fig.add_subplot(212)] 
axs[0].plot(t,ya,t,y,t, y_unpert, linestyle="--",label='demonstration')







