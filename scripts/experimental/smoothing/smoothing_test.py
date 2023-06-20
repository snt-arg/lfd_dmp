#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:29:12 2023

@author: abrk
"""
#%%
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d


with open('0.pickle', 'rb') as file:
    demonstration = pickle.load(file)
  
#%%  
  
joint_trajectory = demonstration.joint_trajectory

n_time_steps = len(joint_trajectory.points)
n_dim = len(joint_trajectory.joint_names)

ys = np.zeros([n_time_steps,n_dim])
ts = np.zeros(n_time_steps)

for (i,point) in enumerate(joint_trajectory.points):
    ys[i,:] = point.positions
    ts[i] = point.time_from_start.to_sec()


#%%

pose_trajectory = demonstration.pose_trajectory

n_time_steps = len(joint_trajectory.points)

positions = np.zeros((n_time_steps, 3))
orientations = np.zeros((n_time_steps, 4))

for (i,point) in enumerate(pose_trajectory.points):
    positions[i,:] = [point.pose.position.x, point.pose.position.y, point.pose.position.z]
    orientations[i,:] = [point.pose.orientation.x, point.pose.orientation.y, point.pose.orientation.z, point.pose.orientation.w] 
    
#%%

def spline(ts, ys):
    splines = []
    for dof in range(ys.shape[1]):
        spline = CubicSpline(ts, ys[:, dof])
        splines.append(spline)

    # Evaluate velocities and accelerations at each time step
    ys_spline = np.zeros_like(ys)  # Array to store spline version of ys
    yds = np.zeros_like(ys)   # Array to store velocities
    ydds = np.zeros_like(ys)  # Array to store accelerations

    for i, t in enumerate(ts):
        for dof in range(ys.shape[1]):
            ys_spline[i, dof] = splines[dof](t)  # Evaluate spline version of ys
            yds[i, dof] = splines[dof](t, 1)  # Evaluate velocity
            ydds[i, dof] = splines[dof](t, 2)  # Evaluate acceleration
    
    return ys_spline, yds, ydds
        
#%%

# Sample trajectory
ts_high = ts  # Time stamps at 20 Hz
ys_high = ys # 7 DOF waypoints at 20 Hz

# Define the desired time stamps at the lower frequency (e.g., 5 Hz)
desired_frequency = 0.5  # Hz
desired_duration = ts_high[-1]
desired_num_samples = int(desired_duration * desired_frequency) + 1
ts_low = np.linspace(0, desired_duration, desired_num_samples)

# Downsample the ys array using interpolation
ys_low = np.zeros((len(ts_low), ys_high.shape[1]))
for i in range(ys_high.shape[1]):
    interpolator = interp1d(ts_high, ys_high[:, i])
    ys_low[:, i] = interpolator(ts_low)
#%%

ys_spline_low, yds_low, ydds_low = spline(ts_low, ys_low)


#%%

from pykalman import KalmanFilter


# Define the Kalman filter model
dim_state = 7  # Dimension of the state (same as the number of DOFs)
dim_obs = 7    # Dimension of the observation (same as the number of DOFs)

# Define the state transition matrix (identity matrix as we assume constant velocity)
transition_matrix = np.eye(dim_state)

# Define the observation matrix (identity matrix as we directly observe the DOFs)
observation_matrix = np.eye(dim_obs)

# Define the initial state and initial covariance matrix
initial_state_mean = np.zeros(dim_state)
initial_state_covariance = np.eye(dim_state)

# Define the process noise covariance matrix (assumed to be diagonal)
process_noise_covariance = np.eye(dim_state) * 1e-4

# Define the observation noise covariance matrix (assumed to be diagonal)
observation_noise_covariance = np.eye(dim_obs) * 1e-2

# Create the Kalman filter object
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    observation_covariance=observation_noise_covariance
)

# Set the process noise covariance matrix
kf.transition_covariance = process_noise_covariance

# Initialize arrays to store velocity and acceleration estimates
yds = np.zeros_like(ys)     # Array to store velocities
ydds = np.zeros_like(ys)    # Array to store accelerations

# Perform Kalman filtering and estimate velocities and accelerations
for i in range(len(ts)):
    if i == 0:
        # Use initial state as the first observation
        state_mean, state_covariance = kf.initial_state_mean, kf.initial_state_covariance
    else:
        # Predict the next state
        state_mean, state_covariance = kf.filter_update(
            filtered_state_mean, filtered_state_covariance
        )

    # Update the filter with the current observation
    filtered_state_mean, filtered_state_covariance = kf.filter_update(
        state_mean, state_covariance, ys[i]
    )

    # Extract the estimated velocity and acceleration from the state
    yds[i] = filtered_state_mean[1]   # Estimated velocity
    ydds[i] = filtered_state_mean[2]  # Estimated acceleration
