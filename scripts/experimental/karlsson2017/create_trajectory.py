#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:49:57 2022

@author: abrk
"""


#############################  Create Trajectory from karlsson2017 Matlab code

#%%

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dmpbbo.dmp.Trajectory import *



# Params
alpha_x = 1
n_kernel = 15

ct = np.linspace(0, 1, n_kernel)[:,np.newaxis]
cx = np.exp(-alpha_x*ct)
c = cx
d = np.power(np.diff(c,axis=0)*0.55,2)
d = 1/np.append(d, d[-1])[:,np.newaxis]

## Demonstration Trajectory, 1 DOF
dt = 1/250;
n_steps = int(10/dt)
t = np.linspace(0,10,n_steps)[:,np.newaxis]
y_demo = np.fmax(0, -np.sin(2*np.pi*t[0:1000]/2.5))
y_demo = np.concatenate((y_demo, y_demo[-1] * np.ones((50,1))))

# Determine DMP Demonstration
t_end = len(y_demo)*dt
tau = t_end
ydot_demo = np.append(np.diff(y_demo,axis=0),0)[:,np.newaxis]/dt
yddot_demo = np.append(np.diff(ydot_demo, axis=0), 0)[:,np.newaxis]/dt


# Create trajectory DMP style
ts = np.linspace(0,(y_demo.size)*dt,y_demo.size)
trajectory = Trajectory(ts,y_demo,ydot_demo,yddot_demo)


#%%
############################ Create Trajectory from ROS-based demos

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

filename = "/home/abrk/ws_moveit/src/lfd_interface/data/demonstrations/testdemo_pickplace1.pickle"

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

trajectory = Trajectory(ts, positions)

#%% Modify trajectory

trajectory = Trajectory(ts, positions[:,1][:,np.newaxis])
