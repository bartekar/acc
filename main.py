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
dt= 0.05

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

u[1,:600]= 4.5
u[1,600:700]= np.linspace(4.5, 3.0, 100)
u[1,700:]= 3.0

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

# %% 5) kpis

def calc_kpi_crash(s_leader, s_follower):
    kpi= np.zeros_like(s_leader)
    dist_prev= s_leader[0] - s_follower[0]
    for k in range(1, s_leader.shape[0]):
        dist_now= s_leader[k] - s_follower[k]
        if np.sign(dist_now) != np.sign(dist_prev):
            kpi[k]= 1
        dist_prev= dist_now
    return kpi

# plt.plot(calc_kpi_crash(X[0,:], X[2,:]))


def calc_safety_dist(v):
    v= np.abs(v) # todo: DO THIS RIGHT
    if v >= 0:
        min_dist= 1 # 1m
        safety_dist= v*1.8 # *3.6 -> km/h, dann halber tachoabstand
        safety_dist+= min_dist
        return safety_dist
    else:
        raise Exception('safety-distance for neg velocity is undefined')
        

#   A kpi
#   |
# 1 *
#   | *
#   |   *                       *
#   |     *                 *
#   |       *           *
# 0 +---------*--*--*--------- > distance between vehicles
#   0         d_s   d_i
# 
# safety distance is defined as d_s = v*3.6/2.0 +eps
# ignorance window is the range from d_S to d_i= (1+i_win)*d_s
# from i on there is another penalty, that increases with higher distance
# 
def calc_kpi_distance(s_leader, s_follower, v_follower, ignorance_win=0.3, penalty_slope=0.2):
    kpi= np.zeros_like(s_leader)
    dist_vehicles= s_leader - s_follower
    for k in range(0, s_leader.shape[0]):
        safety_dist= calc_safety_dist( v_follower[k] )
        ignore_dist= (1+ignorance_win) * safety_dist
        
        if dist_vehicles[k] > ignore_dist:    # right of the diagram
            excess_dist= dist_vehicles[k] - ignore_dist
            kpi[k]= excess_dist/safety_dist *penalty_slope
        elif dist_vehicles[k] > safety_dist:  # inside the window of ignorance
            kpi[k]= 0.0
        elif dist_vehicles[k] >= 0.0:         # diving into safety distance
            kpi[k]= 1.0 - dist_vehicles[k]/safety_dist
        elif dist_vehicles[k] > -3.0:         # inside the other vehicle...
            kpi[k]= 1.0
        elif dist_vehicles[k] <= -3.0:        # "follower" is in front of the "leading vehicle" -> this is fine
            kpi[k]= 0.0
        else:
            raise Exception('this should not have happened')
    return kpi

# plt.plot(calc_kpi_distance(X[0,:], X[2,:], X[3,:]))

# unfinished... is it necessary?
def calc_kpi_jerk(a, dt, j_dangerous):
    j= np.zeros_like(a)
    j[1:]= (a[1:]-a[:-1]) /dt
    # j/= a_max
    return j

# plt.plot(calc_kpi_jerk(u[1,:], dt, j_dangerous=4.5))


# %% 6) controller design through look-up-tables

# input: 
    # consider distances discretized like this: -inf < 0 < 0.5 < d_s < d_s/2+d_i/2 < d_i < d_i*1.5 < inf -> dim=7
    # assume v-leader = const -> ignore
    # divide velocity into intervals of length 5: -inf < 0 < 5 < 10 < 15 < ... < 35 < inf -> dim=9
# output:
    # output of the controller is a float representing the acceleration in the range -4.5 to 4.5

def __dist_2_idx__(d, v, i_win=0.3):
    safety_dist= calc_safety_dist(v)
    ignore_dist= (1+i_win)* safety_dist
    if d < 0.0:
        return 0
    elif d < 0.5*safety_dist:
        return 1
    elif d < safety_dist:
        return 2
    elif d < (safety_dist+ignore_dist)/2.0:
        return 3
    elif d < ignore_dist:
        return 4
    elif d < ignore_dist*1.5:
        return 5
    else:
        return 6

def __velo_2_idx__(v):
    if v < 0.0:
        return 0
    elif v > 35:
        return 8
    else:
        return int(v)//5 +1

def create_evo_controller(magnitude=1.0):
    return np.random.randn(7,9) *magnitude

def apply_evo_controller(x, controller):
    dist= x[0]-x[2]
    idx_dist= __dist_2_idx__(dist, x[3])
    idx_velo= __velo_2_idx__(x[3])
    return controller[idx_dist, idx_velo]


# input: 
    # same as above
# output:
    # output of the controller is the expected value of using one of the following accelerations:
    # -4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5 -> dim=7

def __idx_2_action__(idx):
    accel= [-4.5, -2.0, -0.5, 0.0, 0.5, 2.0, 4.5]
    return accel[idx]

def __action_2_idx__(action):
    if action < -3.25:
        return 0
    elif action < -1.25:
        return 1
    elif action < -0.25:
        return 2
    elif action <  0.25:
        return 3
    elif action <  1.25:
        return 4
    elif action < 3.25:
        return 5
    else:
        return 6

def create_rl_controller(bias= 0.0):
    return np.zeros((7,9,7)) +bias

# epsilon greedy
def apply_rl_controller(x, controller, eps):
    if np.random.rand(1)[0] > eps:
        dist= x[0]-x[2]
        idx_dist= __dist_2_idx__(dist, x[3])
        idx_velo= __velo_2_idx__(x[3])
        values= controller[idx_dist, idx_velo, :]
        best_value_idx= np.argmin(values)
        return __idx_2_action__(best_value_idx)
    else:
        return __idx_2_action__( np.random.randint(0,7) )

def adjust_rl_controller(x, action, new_value, controller):
    dist= x[0]-x[2]
    idx_dist= __dist_2_idx__(dist, x[3])
    idx_velo= __velo_2_idx__(x[3])
    idx_action= __action_2_idx__(action)
    old_value= controller[idx_dist, idx_velo, idx_action]
    controller[idx_dist, idx_velo, idx_action]= 0.6*old_value + 0.4*new_value
    return controller


def create_dummy_controller():
       # v_follower  -inf  0   5    10   15   20   25   30   35   inf
    return np.array([[ 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],  # in front of leading
                     [-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5,-4.5],  # critical behind leading
                     [ 4.5, 4.5, 2.5, 0.5, 0.5, 0.5, 0.5,-0.5,-2.5],  # approaching leading
                     [ 4.5, 4.5, 4.5, 4.5, 4.5, 2.5, 2.5, 0.5, 0.5],  # inside ignore-win, first half
                     [ 4.5, 4.5, 4.5, 4.5, 2.5, 2.5, 2.5, 2.5, 2.5],  # inside ignore-win, second half
                     [ 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2.5, 2.5, 2.5],  # a little behind
                     [ 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]]) # far behind

# ctrl= create_rl_controller()
# adjust_rl_controller([0,0,0,0], -4.5, 1.0, ctrl)

# %% 7) Testing a controller

def simulate(ctrl_fun, trigger_period=1):
    # initialize state space
    X[0,0]= dist0
    X[1,0]= v0
    X[2,0]= 0
    X[3,0]= v1

    u= np.zeros((2,sim_length))
    u[0,:]= v0*r # acceleration for leading. This value is set such that the velocity does not change (terminal velocity)
    # acceleration for follower stays zero and is changed during simulation

    for t in range(0,sim_length-1):
        if t%trigger_period == 0:
            u[1,t]= ctrl_fun(X[:,t])
        else:
            u[1,t]= u[1,t-1]
        X[:2,[t+1]]= (A@X[:2,[t]] +b*u[0,t]) *dt + X[:2,[t]]
        X[2:,[t+1]]= (A@X[2:,[t]] +b*u[1,t]) *dt + X[2:,[t]]
    
    reward= 99*calc_kpi_crash(X[0,:], X[2,:]) + calc_kpi_distance(X[0,:], X[2,:], X[3,:])
    return reward, u[1,:]


# ctrl= create_dummy_controller()
# ctrl_fun= lambda x: apply_evo_controller(x, ctrl)
ctrl_fun= lambda x: apply_rl_controller(x, ctrl, 0.0)
reward, _= simulate(ctrl_fun, trigger_period=1)

print(np.sum(reward))

# %% 8) optimizing using evolutionary algorithms

def evo_search(num_generations= 20, desired_improvement_rate=0.2):
    # final_mutation_rate = decay**num_gen
    # log10(fin_mutate) = log10(decay) * num_gen
    # log10(fin_mutate)/ num_gen = log10(decay)
    decay= np.exp( np.log(0.01)/ num_generations )
    pop_size=20
    num_parents= 4
    num_children= pop_size//num_parents -1
    
    mutation_rate= 1.0
    
    kpi_stats= {'best':[], 'mean':[], 'std':[]}
    
    pop= []
    for k in range(pop_size):
        pop.append( {'ctrl': create_evo_controller(magnitude=1.0), 'my_fitness': np.inf, 'parent_fitness': np.inf} )
    
    for k in range(num_generations):
        print('gen', k, 'of', num_generations)
        # evaluate fitness
        for l in range(pop_size):
            ctrl= pop[l]['ctrl']
            fitness, _= simulate( lambda x: apply_evo_controller(x, ctrl) )
            pop[l]['my_fitness']= np.sum( fitness )
        
        # select sort according to fitness (smallest fitness is first element)
        pop.sort(key= lambda x: x['my_fitness'])
        
        # save status for later evaluation
        gen_fitness= [x['my_fitness'] for x in pop]
        kpi_stats['best'].append(gen_fitness[0])
        kpi_stats['mean'].append( np.mean(gen_fitness) )
        kpi_stats['std'].append( np.std(gen_fitness) )
        
        # adapt mutation rate
        if k > 1:
            # how many instances behaved better then their parent?
            num_better= 0
            for p in pop:
                if p['my_fitness'] < p['parent_fitness']:
                    num_better+= 1
            improvement_rate= num_better/pop_size
            if improvement_rate > desired_improvement_rate:
                mutation_rate/= decay
            else:
                mutation_rate*= decay
        
        # generate new population
        new_pop= []
        for l in range(num_parents):
            # create a "clone" of the parent
            child= {'ctrl': pop[l]['ctrl']}
            child['parent_fitness']= pop[l]['my_fitness']
            child['my_fitness']= pop[l]['my_fitness']
            new_pop.append(child)
            for m in range(num_children):
                ctrl= pop[l]['ctrl'] + create_evo_controller(mutation_rate)
                ctrl[np.where(ctrl >  4.5)] =  4.5
                ctrl[np.where(ctrl < -4.5)] = -4.5
                child= {'ctrl': ctrl }
                child['parent_fitness']= pop[l]['my_fitness']
                child['my_fitness']= np.inf
                new_pop.append(child)
        pop= new_pop
    return kpi_stats, pop[0]['ctrl']


kpi_stats, best_ctrl= evo_search(num_generations=20)

# %% 9) optimizing using reinforcement learning

def rl_search(num_runs=2):
    trigger_period= 4
    trigger_length= int( trigger_period/dt )
    
    num_runs= 1100
    prob_decay= 0.01
    prob_decay_time= 10
    prob_exploration= 1.0 + prob_decay
    
    gamma= np.exp( np.log(0.1)/ 3 )
    
    kpi_stats= []
    
    def kpi_2_value(kpi):
        values= np.zeros_like(kpi)
        values[-1]= kpi[-1]
        for k in range(len(values)-2,-1,-1):
            values[k]= kpi[k] + gamma*values[k+1]
        return values
    
    ctrl= create_rl_controller()
    
    for k in range(num_runs):
        if (k % prob_decay_time) == 0:
            prob_exploration-= prob_decay
            print('Run number', k, 'of', num_runs)
        ctrl_fun= lambda x: apply_rl_controller(x, ctrl, prob_exploration)
        reward, actions= simulate(ctrl_fun, trigger_period=trigger_length)
    
        # remember kpis to compare training
        kpi_stats.append(np.sum(reward))    
    
        # use only true actions to do learning
        actions= actions[0:-1]
        actions= actions[::trigger_length]
        reward= reward[0:-1]
        reward= reward.reshape((-1,trigger_length))
        reward= np.sum( reward, axis=1 ) /trigger_length
        
        values= kpi_2_value(reward)
        
        # update controller
        for l in range( len(actions) ):
            adjust_rl_controller( X[:,l*trigger_length], actions[l], values[l], ctrl )
    
    return kpi_stats, ctrl

kpi_stats, ctrl= rl_search()
plt.clf()
plt.plot(kpi_stats)
plt.grid()
print(kpi_stats[-1])

# todos:
    # reinforcement learning
    # braking VS driving backwards

# bugs:
    # negative velocities -> safety-distance
    # very high accelerations (1.5g)


# bias problem
# lern geschwindigkeit
# value VS optimal move
# problematisches ueberschreiben der falschen zahlen
# reproduzierbarkeit
# wahl der hyper-parameter?
# initial conditions, hyperparameters, ...

# more diverce acceleration for RL
# discretize evo result
# make it robust
# different velocity for leading vehicle
# reproducability (train several times)
# python trainign speed -> profiling -> optimize for speed
# learning speed for more complex LUTs
# starting from standstill
# not driving backwards...
# dynamic behavior of leading vehicle
# jerk penalty

