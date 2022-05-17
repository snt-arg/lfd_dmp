#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:15:28 2022

@author: abrk
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

from dmpbbo.dmp.dmp_plotting import *
from dmpbbo.dmp.Dmp import *
from dmpbbo.dmp.Trajectory import *
from dmpbbo.functionapproximators.FunctionApproximatorLWR import *


tau = 0.8
n_dims = 1
n_time_steps = 81

y_init = np.array([0.0])
y_attr = np.array([1.5])

ts = np.linspace(0, tau, n_time_steps)

# y_yd_ydd_viapoint = np.array([2.0, 0.0, 0.0])
# viapoint_time = 0.6*ts[-1]
# traj = Trajectory.generatePolynomialTrajectoryThroughViapoint(
#         ts, y_init, y_yd_ydd_viapoint, viapoint_time, y_attr)

traj = Trajectory.generateMinJerkTrajectory(ts, y_init, y_attr)

plt.plot(ts, traj.ys_)
plt.plot(ts, traj.ydds_)

function_app = [FunctionApproximatorLWR(10)]
name = "Dmp"
dmp_type='KULVICIUS_2012_JOINING'

dmp = Dmp.from_traj(traj, function_app, name, dmp_type, "G_MINUS_Y0_SCALING")

dmp.set_tau(tau_exec)

tau_exec = 1.2
n_time_steps = 121
ts = np.linspace(0,tau_exec,n_time_steps)

dt = ts[1]
xs_step = np.zeros([n_time_steps,dmp.dim_])
xds_step = np.zeros([n_time_steps,dmp.dim_])

(xs_step[0,:],xds_step[0,:]) = dmp.integrateStart()

for tt in range(1,n_time_steps):
    (xs_step[tt,:],xds_step[tt,:]) = dmp.integrateStep(dt,xs_step[tt-1,:]); 


fig = plt.figure(1)
axs = [ fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133) ]
lines = plotTrajectory(traj.asMatrix(),axs)
plt.setp(lines, linestyle='-',  linewidth=4, color=(0.8,0.8,0.8), label='demonstration')

traj_reproduced = dmp.statesAsTrajectory(ts,xs_step,xds_step)
lines = plotTrajectory(traj_reproduced.asMatrix(),axs)
plt.setp(lines, linestyle='--', linewidth=2, color=(0.0,0.0,0.5), label='reproduced')

plt.legend()
fig.canvas.set_window_title('Comparison between demonstration and reproduced') 