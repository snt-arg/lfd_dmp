#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:15:50 2022

@author: abrk
"""
#%%

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt



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
dt = 1/250;
n_steps = int(10/dt)
t = np.linspace(0,10,n_steps)[:,np.newaxis]
y_demo = np.fmax(0, -np.sin(2*np.pi*t[0:1000]/2.5))
y_demo = np.concatenate((y_demo, y_demo[-1] * np.ones((50,1))))

# Determine DMP Demonstration
t_end = len(y_demo)*dt
tau = t_end/3
p = len(y_demo)
g = y_demo[-1]



# Determine the weight (traj2w)

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
    