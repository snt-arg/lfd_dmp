#####!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:28:39 2022

@author: abrk
"""

# Import stuff
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from dmpbbo.dmp.dmp_plotting import *
from dmpbbo.dmp.Dmp import *
from dmpbbo.dmp.Trajectory import *
from dmpbbo.functionapproximators.FunctionApproximatorLWR import *


## Demonstration Trajectory, 1 DOF
dt = 1/250;
n_steps = int(10/dt + 1)
t = np.linspace(0,10,n_steps)
traj = np.fmax(0, -np.sin(2*np.pi*t[0:1000]/2.5))
traj = np.concatenate((traj, traj[-1] * np.ones(50)))

t_end = len(traj)*dt
tau = t_end/3
ts = np.linspace(0,tau,len(traj))

traj = np.expand_dims(traj, axis=1)
demo_traj = Trajectory(ts, traj)


# fig = plt.figure(0)
# plt.plot(traj)



function_apps = []
function_apps.append(FunctionApproximatorLWR(15))

dmp_type='IJSPEERT_2002_MOVEMENT'

dmp = Dmp.from_traj(demo_traj, function_apps, dmp_type, forcing_term_scaling="G_MINUS_Y0_SCALING")


#Integrate

tau_exec = 2100*dt
n_steps = int(tau_exec/dt + 1)
ts = np.linspace(0,tau_exec,n_steps)

xs_step = np.zeros([n_steps,dmp.dim_])
xds_step = np.zeros([n_steps,dmp.dim_])

(x,xd) = dmp.integrateStart()
xs_step[0,:] = x;
xds_step[0,:] = xd;

for tt in range(1,n_steps):
    (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 
    
    
    
# Plotting

fig = plt.figure(1)
axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 

lines = plotTrajectory(demo_traj.asMatrix(),axs)
plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')

traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)
lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')

plt.legend()
fig.canvas.set_window_title('Comparison between demonstration and reproduced')



# Purturbed Integration

e = 0
kc = 10000;
tau_adapt = tau*(1+(kc*e^2))

