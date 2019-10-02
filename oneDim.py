# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:55:33 2019

@author: Sherin
"""

import numpy as np
import time
import scipy.signal as sim
import scipy.integrate as sol
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r'C:\Users\sheri\Desktop\Sherins Map All Things\Research\Active Inference\SSAI\active_inference')

import ssai as ai


# Control of one dimensional system using AI

# System parameters:
m = 1e3
d = 8e2

a = np.matrix([[-d/m]])
b = np.matrix([[1/m]])
c = np.matrix([[1]])

system = sim.StateSpace(a,b,c,0)
n = np.size(a)
q = np.size(c)

x_eq = 40

# Tuning parameters:
a_mu  = 5
a_u   = 1e7
k     = 2
Cw    = 10
Cz    = 10
gamma = 32

# Construct time vector:
T  = 20
dt = 0.1
t  = np.arange(0,T+dt,dt)
N  = np.size(t)

# Construct noise signals:
w = ai.makeNoise(Cw,gamma,t)
z = ai.makeNoise(Cz,gamma,t)

# Set up the Active Inference agent:
agent = ai.activeInferenceAgent(k,gamma,Cw,Cz,a_mu,a_u,system)

# Define the prior variable (default is 0):
muref = np.matrix([[x_eq],[0]])
K     = np.matrix([0])
agent.setPolePlacement(K)
xi   = (agent.gm.D-agent.gm.A).dot(muref)
xi   = np.matlib.repmat(xi,1,N)
agent.xi = xi

# Simulate the closed loop:
s0 = np.ndarray.flatten(np.zeros([1,n*(1+k) + q])) # initial state
tic = time.time()
st = sol.odeint(ai.dynamics,s0,t,args=(dt,w,z,system,agent))
tsim = time.time() - tic

print('Simulation time: ' + "%.2f" % tsim + ' sec')

# Calculate free energy of simulation:
x  = np.matrix(st[:,0:n]).T
y  = system.C.dot(x) + z
mu = np.matrix(st[:,n:n*k+1]).T
u  = np.matrix(st[:, n*k+1:]).T

F = agent.getFreeEnergy(mu,y,xi)

# Plot results:
fig, axes = plt.subplots(2,2)
fig.suptitle('Active Inference on first order system')

axes[0,0].plot(t,x.T,label='State')
axes[0,0].plot(t,mu[0,:].T,label='Belief')
axes[0,0].set_ylabel('System state x')
axes[0,0].legend()
axes[0,0].grid(1)
axes[1,1].plot(t,w.T)
axes[1,1].plot(t,z.T)
axes[1,1].set_xlabel('time [s]')
axes[1,1].set_ylabel('Noise')
axes[1,1].grid(1)
axes[0,1].plot(t,u.T)
axes[0,1].ticklabel_format(style='sci',scilimits=(0,0))
axes[0,1].set_ylabel('Control u')
axes[0,1].grid(1)
axes[1,0].semilogy(t,F.T)
axes[1,0].set_xlabel('time [s]')
axes[1,0].set_ylabel('Free energy')
axes[1,0].grid(1)

    


