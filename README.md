# Active Inference for Robotics

**Note:**
This readme will be elaborated once the code has been committed.

## Description
Active Inference, a result of the Free-Energy principle, has been proposed by K. Friston to explain perception and action in the same framework. There have been only a few attempts to implement active inference for control in robotics. This project will focus on implementing Active Inference for robotic systems. The idea is to increase complexity of controlled systems over the course of time. We will also study the relation between Active Inference and existing algorithms, specifically uncovering the benefits of generalised motions. 

This project is part of the research line on Prediction Error Minimisation lead by Prof. dr. ir. Martijn Wisse to apply Active Inference in robotics.

Project contact: Sherin Grimbergen (sheringrimbergen@hotmail.com)


### Keywords
_Active inference, Free Energy Principle, Generalised motions, Optimal Control, Robot Control._


## Project Goals:
1. Perform Active Inference on different types of systems (LTI state space, nonlinear, multibody, etc.)
2. Analyze performance and algorithm compared to optimal control algorithms (e.g. LQG)
3. Study the effect of adding generalised motions
4. Implement the [DEM algorithm](https://doi.org/10.1016/j.neuroimage.2008.02.054) (check [this](https://github.com/uclyyu/DemActiveInference/blob/master/dynamic/vid.py) out too).

## How to use:
The examples folder contains a bunch of simulations that can be performed. Just read and run some of these files to get acquainted with the code. Currently, there are two main functionalities in the project, from the `ssai.py` file. The first is the generation of non-Markovian noise. An example:

```python
Cw    = 10 # covariance matrix
gamma = 32 # roughness parameter

# Construct time vector:
T  = 20
dt = 0.05
t  = np.arange(0,T+dt,dt)

# Construct noise signals:
w = ssai.makeNoise(Cw,gamma,t)
```

The second and most important part is the construction of an Active Inference agent, for which we use a class `activeInferengeAgent()`:

```python
# Tuning parameters:
a_mu  = 5       # perception learning rate
a_u   = 1e7     # action learning rate
k     = 2       # embedding order
Cw    = 10      # state noise covariance
Cz    = 10      # observation noise covariance
gamma = 32      # roughness parameter

# Set up the Active Inference agent:
agent = ssai.activeInferenceAgent(k,gamma,Cw,Cz,a_mu,a_u,system)
```

In here *system* is a state space model as can be obtained from `scipy.linalg.StateSpace(a,b,c,d)`.

## Status

### In progress
- KF vs DEM simulation
- Implementing the algorithm for planar 2DOF arm  

### Completed
- First simulation is working `oneDim.py`

## References
- First Active Inference paper by Friston et al.: [link](https://doi.org/10.1007/s00422-010-0364-z)
- Tutorial on Free Energy Principle by Bogazc: [link](https://doi.org/10.1016/j.jmp.2015.11.003)
- Tutorial on Active Inference by Grimbergen et al.: [link](http://dx.doi.org/10.1098/rsif.2016.0616)

<!--
### Usefull semantics:
Find more information [here](https://about.gitlab.com/handbook/product/technical-writing/markdown-guide/)

Key function from SPM: `spm_adem`

```python
v =  np.array([1 0],[4 3])
def hello():
    return []
```

**NOTE:**
*M(i).f  = dx/dt = f(x,v,P)*    {inline function, string or m-file}
`SevDofthreeDv1simpleROS_spm_fx_robot_dem_reach_v2.m`
implements de controller in **2.18** in *Pio-Lopez* and the simulation in ROS to obtain the position and orientation of the robot links.

* In *Friston*, `spm_DEM(DEM)` only takes `DEM`, so the target position is coded within DEM as the causal states.

-->
