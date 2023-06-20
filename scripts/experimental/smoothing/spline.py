#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:29:12 2023

@author: abrk
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Sample trajectory data
t = np.array([0, 1, 2, 3, 4])  # Time
x = np.array([2, 4, 7, 11, 16])  # X positions
y = np.array([3, 6, 9, 12, 15])  # Y positions
z = np.array([1, 2, 4, 8, 16])  # Z positions

# Create the spline interpolation functions for each coordinate
spline_x = CubicSpline(t, x)
spline_y = CubicSpline(t, y)
spline_z = CubicSpline(t, z)

# Differentiate the spline functions to obtain velocities and accelerations
dt = 0.01  # Time step for differentiation
t_diff = np.arange(t[0], t[-1], dt)  # New time array for differentiation
velocities = np.array([spline_x(t_diff, 1), spline_y(t_diff, 1), spline_z(t_diff, 1)]).T
accelerations = np.array([spline_x(t_diff, 2), spline_y(t_diff, 2), spline_z(t_diff, 2)]).T

# Print the results
print("Velocities:")
print(velocities)
print("Accelerations:")
print(accelerations)


# Plot original trajectory
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.plot(t, x, 'ro-', label='X')
plt.plot(t, y, 'go-', label='Y')
plt.plot(t, z, 'bo-', label='Z')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Original Trajectory')
plt.legend()

# Plot interpolated trajectory
plt.subplot(2, 2, 2)
plt.plot(t_diff, spline_x(t_diff), 'r-', label='X')
plt.plot(t_diff, spline_y(t_diff), 'g-', label='Y')
plt.plot(t_diff, spline_z(t_diff), 'b-', label='Z')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Interpolated Trajectory')
plt.legend()

# Plot velocities
plt.subplot(2, 2, 3)
plt.plot(t_diff, velocities[:, 0], 'r-', label='X')
plt.plot(t_diff, velocities[:, 1], 'g-', label='Y')
plt.plot(t_diff, velocities[:, 2], 'b-', label='Z')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocities')
plt.legend()

# Plot accelerations
plt.subplot(2, 2, 4)
plt.plot(t_diff, accelerations[:, 0], 'r-', label='X')
plt.plot(t_diff, accelerations[:, 1], 'g-', label='Y')
plt.plot(t_diff, accelerations[:, 2], 'b-', label='Z')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Accelerations')
plt.legend()

plt.tight_layout()
plt.show()