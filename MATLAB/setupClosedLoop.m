function AIss = setupClosedLoop(process, agent)
%% This function constructs the closed loop state space model
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     22-06-2018
%
% INPUTS:
% process   = Generative Process (State Space Model)
% agent     = Generative Model   (Structure):
%
% agent.A      =   A matrix of generative model
% agent.C      =   C matrix of generative model
% agent.D      =   Derivative operator
% agent.G      =   Forward Model of agent
% agent.Piz    =   Precision matrix output noise
% agent.Piw    =   Precision matrix state noise
% agent.kappa  =   Perception learning rate
% agent.rho    =   Action learning rate
% agent.p      =   Embedding order
%
% OUTPUTS:
% AIss          =   State Space Model of the closed loop system
%
% NOTE:
% Verify whether used has supplied a continuous system model:
assert(process.Ts == 0,'You cannot use a discrete state space model');
% Make sure the user does not require generalized measurements, as these
% have not been implemented yet:
assert(~agent.ygen,'No functionality yet for generalized measurements, sorry!');

%% Pre-processing
% Extract process matrices:
A = process.A;
B = process.B;
C = process.C;

% Unpack generative model:
Atilde = agent.A;
Ctilde = agent.C;
D      = agent.D;
Ghat   = agent.G;
p      = agent.p;
rho    = agent.rho;
kappa  = agent.kappa;
Piz    = agent.Piz;
Piw    = agent.Piw;

% Dimensions:
n = size(A,2);  % state
m = size(B,2);  % input
q = size(C,1);  % output

%% Processing:
M = D - kappa*(D-Atilde).'*Piw*(D-Atilde) - kappa*Ctilde.'*Piz*Ctilde;

% The state of the model is [x mu u], hence the 3x3 partitioning of Acl:
Acl = [A                     zeros(n,n*p)        B;...
       kappa*Ctilde.'*Piz*C  M                   zeros(n*p,m);...
      -rho*Ghat'*Piz*C        rho*Ghat'*Piz*Ctilde zeros(m,m)];

% The inputs are [xi w z], hence the 3x3 partitioning of Bcl: 
Bcl = [zeros(n,n*p)            eye(n)        zeros(n,q);...
       kappa*(D-Atilde).'*Piw  zeros(n*p,n)  kappa*Ctilde.'*Piz;... 
       zeros(m,n*p)            zeros(m,n)   -rho*Ghat'*Piz];

% For inspection purposes, we choose to measure all signals:
Ccl = eye(size(Acl));

% There is no direct feedthrough of any signal:
Dcl = 0;

% Put the four matrices in a state space sturcture (continuous):
AIss = ss(Acl,Bcl,Ccl,Dcl);

end

