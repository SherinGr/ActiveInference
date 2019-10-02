# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:30:08 2019

@author: Sherin

In this file we compare Generalised Filtering and the Kalman Bucy Filter under
the assumption of Markovian noise and a linear system. The two algorithms should,
according to Friston (I believe) be equal. If this simulation shows that, there
must be a mathematical proof for this.

SIMULATION IS NOT FINISHED

"""

import numpy as np
import time
import scipy.signal as sim
import scipy.integrate as sol
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r'C:\Users\sheri\Desktop\Sherins Map All Things\Research\Active Inference\SSAI\active_inference')

import ssai as ai

# System parameters:
m = 1e1
k = 1e2
d = 4e1

a = np.matrix([[0,1],[-d/m, -k/m]])
b = np.matrix([[0],[1/m]])
c = np.matrix([[1,0]])

system = sim.StateSpace(a,b,c,0)
n = a.shape[0] # state dimension
q = c.shape[0] # observation dimension

# Desired equilibrium
x_ref = np.matrix([[40],[0]])

# Tuning parameters:
a_mu  = 1e1
a_u   = 1e4
k     = 3
Cw    = np.matrix([[10,0],[0,10]])
Cz    = 10
gamma = 2e4

# Construct time vector:
T  = 30
dt = 0.1
t  = np.arange(0,T+dt,dt)
N  = np.size(t)

# Construct noise signals:
w = ai.makeNoise(Cw,gamma,t)
z = ai.makeNoise(Cz,gamma,t)

# Set up the Active Inference agent:
agent = ai.activeInferenceAgent(k,gamma,Cw,Cz,a_mu,a_u,system)

# Define the prior variable:
K     = np.matrix([[0,0],[0,0]])
agent.setPrior(x_ref,K)

"Generate data:"
def ss_dyn(x,t,dt,w,system):
    i = int(round(t/dt))
    
    x = np.matrix(x).T
    dx = system.A.dot(x) + w[:,i:i+1]
    
    return np.squeeze(np.array(dx))

x0 = np.ndarray.flatten(np.zeros([1,n])) # initial state
xt = sol.odeint(ss_dyn,x0,t,args=(dt,w,system))
yt = system.C.dot(xt.T) + z

"Perform simulation of Kalman Bucy Filter:"
def kf_dyn(s,t,system):
    # s contains xhat and K
    
    dx = Ax


"Perform simulation of Generalised Filter:"



# Plot results:

fig = plt.plot(t,xt[:,1],label='State')
fig.set_ylabel('System state x')
fig.legend()
fig.grid(1)
    


