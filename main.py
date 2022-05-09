# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:41:42 2022

@author: kbartecki

Simulation of an ACC
Two vehicles drive on the same lane. Given the velocities of both vehicles and the distance between them, find a good
controller for the acceleration of the follower vehicle to maximize the KPIs:
    - Do not crash into the leading vehicle
    - Do not drive too much into the safety distance
    - Keep yourself close to the safety distance of the leading vehicle
    - Do not jerk around^^ (smooth acceleration)    

for the future:
    - different velocities of the leading vehicle
    - different velocities of the leading vehicle
    - emergency stop
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background2')

# %% 1) defining the state space model through ODEs

# user constants

# air resistance [1/s]
r= 0.1
# initial velocity [m/s]
v0= 10.0
# simulation length [s]
T= 30
# simulation step [s], screw around with this parameter too much and you an invalid simulation
dt= 0.05

# automatic generated variables

A= np.array([[0.0, 1.0], 
             [0.0, -r]])
b= np.array([[0.0],
             [1.0]])

sim_length= int(T/dt)+1 # +1 for t=0 and simulation end

X= np.zeros((2,sim_length)) # X is the state space for each time step. First row is driven distance, second row is v.

# initialize state space
X[0,0]= 0
X[1,0]= v0

# %% 2) defining system input

u= np.zeros((1,sim_length))
u[0,:]= 4.5

# %% 3) simulation

for t in range(0,sim_length-1):
    X[:,[t+1]]= (A@X[:,[t]] +b*u[0,t]) *dt + X[:,[t]]


# %% 4) evaluation - does it make sense? 

sim_time= np.linspace(-dt,T,sim_length)

plt.clf()
plt.subplot(211)
plt.title('driven distance')
plt.plot(sim_time, X[0,:])
plt.grid()
plt.ylabel('distance [m]')
plt.xlabel('time [s]')

plt.subplot(212)
plt.title('velocity')
plt.plot(sim_time, X[1,:])
plt.grid()
plt.ylabel('velocity [m/s]')
plt.xlabel('time [s]')











