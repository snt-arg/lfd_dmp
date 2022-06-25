#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:43:26 2022

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

from lfd_interface.msg import DemonstrationMsg


# Read demonstration file

filename = "/home/abrk/ws_moveit/src/lfd_interface/data/demonstrations/testdemo_rviz.pickle"

with open(filename, 'rb') as file:
    demonstration = pickle.load(file)


n_steps = len(demonstration.joint_trajectory.points)
n_dim = len(demonstration.joint_trajectory.joint_names)

positions = np.zeros([n_steps,n_dim])
ts = np.zeros(n_steps)


for (i,point) in enumerate(demonstration.joint_trajectory.points):
    positions[i,:] = point.positions
    ts[i] = point.time_from_start.to_sec()

# Wrap in a dmpbbo Trajectory class

traj = Trajectory(ts, positions)


# # Plotting playground

# fig = plt.figure(0)

# axs = [fig.add_subplot(171), fig.add_subplot(172), fig.add_subplot(173), fig.add_subplot(174)
#        , fig.add_subplot(175), fig.add_subplot(176), fig.add_subplot(177)]

# for (ii,ax) in enumerate(axs):
#     ax.plot(traj.ts_, traj.yds_[:,ii])
    

# play with traj variables

ys = traj.ys_
yds = traj.yds_
ydds = traj.ydds_


traj.yds_[-1,:] = 0
traj.ydds_[-1,:] = 0

#Train DMP

function_apps = []

for i in range(0,n_dim):
    function_apps.append(FunctionApproximatorLWR(10))
    
name='Dmp'
dmp_type='IJSPEERT_2002_MOVEMENT'
# dmp_type='KULVICIUS_2012_JOINING'

dmp = Dmp.from_traj(traj, function_apps, name, dmp_type)

tau_exec = traj.ts_[-1] + 1
n_steps = n_steps + 101
ts = np.linspace(0,tau_exec,n_steps)

( xs_ana, xds_ana, forcing_terms_ana, fa_outputs_ana) = dmp.analyticalSolution(ts)

dt = ts[1]
xs_step = np.zeros([n_steps,dmp.dim_])
xds_step = np.zeros([n_steps,dmp.dim_])

(x,xd) = dmp.integrateStart()
xs_step[0,:] = x;
xds_step[0,:] = xd;
for tt in range(1,n_steps):
    (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 



# Plotting playground

fig = plt.figure(1)
axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ] 

lines = plotTrajectory(traj.asMatrix(),axs)
plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')

traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)
lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')

plt.legend()
fig.canvas.set_window_title('Comparison between demonstration and reproduced') 


fig = plt.figure(2)
ts_xs_xds = np.column_stack((ts,xs_ana,xds_ana))
plotDmp(ts_xs_xds,fig,forcing_terms_ana,fa_outputs_ana)
fig.canvas.set_window_title('Analytical integration') 

fig = plt.figure(3)
ts_xs_xds = np.column_stack((ts,xs_step,xds_step))
plotDmp(ts_xs_xds,fig)
fig.canvas.set_window_title('Step-by-step integration') 


# playing with reproduced trajectory
ys = traj_reproduced.ys_
yds = traj_reproduced.yds_
ydds = traj_reproduced.ydds_


# velocities = np.zeros([n_steps,n_dim])
# accelerations = np.zeros([n_steps,n_dim])

# for i in range(0,n_steps-1):
#     velocities[i,:] = (positions[i+1,:] - positions[i,:])/(ts[i+1] - ts[i])

# velocities[-1,:] = 0

# for i in range(0,n_steps-1):
#     accelerations[i,:] = (velocities[i+1,:] - velocities[i,:])/(ts[i+1] - ts[i])
    
# accelerations[-1,:] = 0

# fig = plt.figure(1)

# axs = [fig.add_subplot(171), fig.add_subplot(172), fig.add_subplot(173), fig.add_subplot(174)
#        , fig.add_subplot(175), fig.add_subplot(176), fig.add_subplot(177)]

# for (ii,ax) in enumerate(axs):
#     ax.plot(ts, accelerations[:,ii])

# plt.plot(ts, positions[:,1])



