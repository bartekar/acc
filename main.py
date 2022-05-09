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
# initial velocity for leading [m/s]
v0= 20.0
# initial velocity for following [m/s]
v1= 25.0
# initial distance between vehicles [m]. Distance is defined as driven-dist(leading) - driven-dist(follower)
dist0= 250
# simulation length [s]
T= 60
# simulation step [s], screw around with this parameter too much and you an invalid simulation
dt= 0.2

# automatic generated variables

A= np.array([[0.0, 1.0], 
             [0.0, -r]])
b= np.array([[0.0],
             [1.0]])

sim_length= int(T/dt)+1 # +1 for t=0 and simulation end

X= np.zeros((4,sim_length)) # X is the state space for each time step.
# First row is driven distance, second row is velocity of the leading vehicle.
# Third row is driven distance, fourth row is velocity of the following vehicle.

# initialize state space
X[0,0]= dist0
X[1,0]= v0
X[2,0]= 0
X[3,0]= v1

# %% 2) defining system input

u= np.zeros((2,sim_length))
u[0,:]= v0*r # acceleration for leading. This value is set such that the velocity does not change (terminal velocity)
u[1,:]= 4.5 # acceleration for follower

# %% 3) simulation

for t in range(0,sim_length-1):
    X[:2,[t+1]]= (A@X[:2,[t]] +b*u[0,t]) *dt + X[:2,[t]]
    X[2:,[t+1]]= (A@X[2:,[t]] +b*u[1,t]) *dt + X[2:,[t]]


# %% 4) evaluation - does it make sense? 

sim_time= np.linspace(-dt,T,sim_length)

plt.clf()
plt.subplot(311)
plt.title('driven distance')
plt.plot(sim_time, X[0,:], label='leading')
plt.plot(sim_time, X[2,:], label='follower')
plt.grid()
plt.legend()
plt.ylabel('distance [m]')

plt.subplot(312)
plt.title('distance between vehicles')
plt.plot(sim_time, np.abs(X[0,:]-X[2,:]))
plt.grid()
plt.ylabel('distance [m]')

plt.subplot(313)
plt.title('velocity')
plt.plot(sim_time, X[1,:], label='leading')
plt.plot(sim_time, X[3,:], label='follower')
plt.grid()
plt.legend()
plt.ylim([0,50])
plt.ylabel('velocity [m/s]')
plt.xlabel('simulation time [s]')





# todos:
    # kpi
    # self tuned PID controller for single v0
    # implement control through state
    # evolutionary algorithms
    # reinforcement learning









