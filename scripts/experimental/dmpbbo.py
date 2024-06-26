#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:04:25 2023

@author: abrk
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


from dmpbbo.dmps.Dmp import Dmp
from dmpbbo.dmps.Trajectory import Trajectory
from dmpbbo.functionapproximators.FunctionApproximatorBSpline import FunctionApproximatorBSpline

from lfd_interface.msg import DemonstrationMsg

from scipy.interpolate import CubicSpline, BSpline




#%%

def create_trajectory(demonstration, index = None):
    n_time_steps = len(demonstration.joint_trajectory.points)
    n_dim = len(demonstration.joint_trajectory.joint_names)

    ys = np.zeros([n_time_steps,n_dim])
    yds = np.zeros([n_time_steps,n_dim])
    ydds = np.zeros([n_time_steps,n_dim])
    ts = np.zeros(n_time_steps)

    for (i,point) in enumerate(demonstration.joint_trajectory.points):
        ys[i,:] = point.positions
        yds[i,:] = point.velocities or None
        ydds[i,:] = point.accelerations or None
        ts[i] = point.time_from_start.to_sec()

    if np.isnan(yds).any(): 
        yds = None
    if np.isnan(ydds).any(): 
        ydds = None
        
    if index is None:
        return Trajectory(ts, ys, yds, ydds)
    else:
        return Trajectory(ts, ys[:,[index]], yds[:,[index]], ydds[:,[index]])


def plot(demo, plan):
    lines, axs = demo.plot()
    plt.setp(lines, linestyle="-", linewidth=4, color=(0.8, 0.8, 0.8))
    # plt.setp(lines, label="demonstration")

    lines, axs = plan.plot(axs)
    plt.setp(lines, linestyle="--", linewidth=2, color=(0.0, 0.0, 0.5))
    # plt.setp(lines, label="reproduced")
    plt.tight_layout()

    plt.legend()
    t = f"Comparison between demonstration and reproduced"
    plt.gcf().canvas.set_window_title(t)

    plt.show()


#%%

filename = "/home/abrk/catkin_ws/src/lfd/lfd_interface/data/demonstrations/smoothfrplace/0.pickle"

with open(filename, 'rb') as file:
    demonstration = pickle.load(file)

index = 2
traj = create_trajectory(demonstration, index)

#%%

function_apps = [FunctionApproximatorBSpline() for _ in range(traj.dim)]
dmp = Dmp.from_traj(traj, function_apps, dmp_type="IJSPEERT_2002_MOVEMENT")    

#%%

# start = [-0.477703, 0.787349, -0.617041, -1.79873, 0.577409, 2.38477, -0.938441]
# goal = [0.755491, 0.493148, 0.043999, -2.10643, 0.0042121, 2.55332, 1.62703]
start = [-1, 0.787349, -0.617041, -1.79873, 0.577409, 2.38477, -1.5]
goal = [0.755491, 0.493148, 0.043999, -2.10643, 0.0042121, 2.55332, 2]
tau_scale = 1

dmp.tau *= tau_scale
if len(start) != 0:
    dmp.y_init = np.array(start[index])
if len(goal) != 0:
    dmp.y_attr = np.array(goal[index])

tau = dmp.tau

#%%

n_time_steps = 100
ts = np.linspace(0,tau,n_time_steps)
dt = ts[1]


(x,xd) = dmp.integrate_start()
xs_step = np.zeros([n_time_steps,x.shape[0]])
xds_step = np.zeros([n_time_steps,x.shape[0]])
f_step = np.zeros([n_time_steps,dmp._dim_y])

xs_step[0,:] = x
xds_step[0,:] = xd

for tt in range(1,n_time_steps):
    f_step[tt,:] = dmp._compute_func_approx_predictions(xs_step[tt-1,dmp.PHASE])
    (xs_step[tt,:],xds_step[tt,:]) = dmp.integrate_step(dt,xs_step[tt-1,:])

new_traj = dmp.states_as_trajectory(ts,xs_step,xds_step)

#%%
plot(traj, new_traj)

#%%

altered_traj = dmp.states_as_trajectory(ts*2,xs_step,xds_step/2)
plot(traj, altered_traj)

#%%
