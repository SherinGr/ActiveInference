# Mindmap: Active Inference Research in Robotics

This file maps out my ideas about the research topics that can be addressed considering the application of Active Inference in robotics. I consider two main branches of research which are:
- **Fixed form** model approaches: If the structure of the system is known, the internal models can be based on the same structure.
- **Free form** model approaches: The models inside the agents are of a free form like neural networks or [kriging](https://en.wikipedia.org/wiki/Kriging)

For both of these approaches there are several high level questions that are relevant:
- What is the benefit of using generalised motions?
- Can we learn the highest level priors or do they require to be hardcoded?
- Will Active Inference outperform existing methods?

## Fixed form models:
There are two major topics to consider in the realm of fixed form Active Inference. First, this approach allows for very detailed research in terms of _mathematical analysis_. Second, this approach allows us to slowly _increase complexity_ of the controlled systems without losing grip on the inner workings of the algorithm. 

Of course, there are many different types of dynamic systems to which Active Inference can be applied. In the free form approach, basically any system can be considered. However, when using fixed form models there are three main types of dynamic system models to consider:
1. Linear state space models
2. Nonlinear state space models
3. POMDP's

With this approach it is very hard to apply the idea of [hierarchical models](https://doi.org/10.1371/journal.pcbi.1000211), because it is unclear what should be present in each layer. This concept lends itself more for the free form models. It is very much comparable to deep neural networks.

### 1. Linear state space models:
This project contains the implementation of Active Inference on LTI state space models. Both in continuous and discrete time. The implementations can be extended or improved in the following ways:
- Add parameter and hyperparameter learning 
- Add causal states *v* (also check whether this makes sense)
- Study the definition of the forward model. There are several approaches, none of which at this point seems to make perfect sense. One approach is to derive the exact mathematically defined construct, which seems very complex. Friston proposes to simply use signs, as this model corresponds to reflex arcs.  
- Change gradient descent to other optimization algorithms, perhaps optimizing the free energy in each iteration. This was an important comment by [F. Oliehoek](https://www.fransoliehoek.net/wp/) during my thesis defence: In typical machine learning algorithms, the action is determined by optimizing the value function at each iteration. The gradient descent in the free energy however, optimizes the free energy over time. This could make it more robust against to disturbances. 

In the linear framework, there is an abundance of tools from control science that allow us to analyse the algorithm in detail. The main interests are to analyse:
- **Optimality**:   Comparison of [DEM](https://doi.org/10.1016/j.neuroimage.2008.02.054) (or the filter of Active Inference) with a Kalman filter. With and without generalised motions.
- **Stability**:    Prove that the free energy is a _Lyapunov function_
- **Performance**:  Effects of generalised motions, in what situations does it make Active Inference _better_ than LQG?
- **Robustness**:   Effects of disturbances, faults (sensors and actuators).

For an easier comparison with existing control methods, it would be convenient if we can seperate the filter and controller dynamics. Namely, the original definition of Active Inference uses the generative model to define track references, which is unconventional. For optimal control, the [separation principle](https://en.wikipedia.org/wiki/Separation_principle) applies

Besides this, it could be interesting to find a _deterministic_ version of the algorithm. It is inherently designed for stochastic systems, or systems involving uncertainty. The main problem when removing noises is that precision matrices tend to infinity, destabilizing the perception and action updates.

### 2. Nonlinear state space models:
A first step towards actual robotic implementations is to make the step from linear to nonlinear models, still with perfect model knowledge. It is interesting to see what the behavior of the controller will be in these settings. Afterwards, the same additions as for the linear model case can be made. Learning of parameters and hyperparameters (which means replicating DEM) can be added once the simpler version works properly and once the DEM algorithm is understood. The latter is the main research topic of [Ajith A. Meera](https://www.tudelft.nl/staff/a.anilmeera/). Addition of the hierarchical framework could be possible, but we should ask ourselves at which level of system complexity it actually starts to make sense to use hierarchical models. It mainly seems interesting because it allows for the definition of high-level priors that are intuitively hardcoded.

Considering robotic arms, a very interesting subject arises. Friston himself is also very excited about this: In robotic arms there is a clear distinction between _configuration space_ (joint angles) and _work space_ (end effector coordinates). We could very easily track dynamic end effector references if we could form a generative model in work space variables that connects the actual end effector with the reference using virtual springs and dampers.

### 3. Partially Observable Markov Decision Processes (POMDP's)
A completely different approach to control is through POMDP's. This model type is often used for (deep) reinforcement learning. Understanding the formulation of Active Inference in this framework (see [here](https://doi.org/10.1162/NECO_a_00912)) brings us closer to possible combinations of Active Inference and these powerful deep learning algoritms. That could be a very feasible approach towards truly intelligent robot control algorithms. Active Inference provides the high level algorithm structure, i.e., _prediction error minimisation_ and the fact that both perception and action are _free energy minimisation_, whereas deep learning algorithms provide the low level implementation of these concepts. 

At the TU Delft, there is nobody yet who understands this formulation, and the literature is very complex. Our best approach would be to collaborate with people who understand POMD's, or to have someone like [Thomas Parr](https://www.ucl.ac.uk/mbphd/current-students-and-alumni/parr-thomas) explain it. There are very little details available about the implementation of algorithm as presented in the papers.

## Free form models:
Intuitively it seems that the true power of Active Inference becomes more apparent as the system to be controlled gets more complex. Where many algorithms will break down, Active Inference will stay functional. This is however merely an intuitive opinion. In all its glory, Active Inference provides an unsupervised learning algorithm that requires only the highest level priors to be hardcoded. If it works in this situation, it is more powerful then all existing machine learning algorithms to date, providing a very efficient way to completely close the loop and to learn controling any type of dynamic system. 

In the Active Inference algorithm, there are two (or three) models that have to be learned by the agent in order to perform control: The state dynamics model, the observation model and - if this is included - the disturbance model. If one or more of these models are very complex or have no proper analytic approximation, the best approach is to use free form models to approximate (and learn) the models. Several interestings approaches are:

- GPR:  Gaussian Process Regression (i.e. kriging). 
- DNN:  Deep neural network
- GAN:  Generative Adversarial Network
- VAE:  Variational Autoencoder

Except for the GPR, these algorithms inherently implement the hierarchical structure. A first implementation in this direction has been studied by K. Ueltzhöffer ([link](10.1007/s00422-018-0785-7))

## Research priorities:
1. Find the exact relations between Active Inference and optimal control
2. Make Active Inference outperform other algorithms in realistic settings
3. Robotic Implementations 
