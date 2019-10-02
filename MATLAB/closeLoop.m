function AIss = closeLoop(process, agent)

%% Pre-processing
% Extract process matrices:
A = process.A;
B = process.B;
C = process.C;

% Unpack generative model:
Atilde = agent.Atilde;
Ctilde = agent.Ctilde;
Der      = agent.Der;
G      = agent.G;
p      = agent.p;
rho    = agent.rho;
kappa  = agent.kappa;
Piz    = agent.Piz;
Piw    = agent.Piw;

% Dimensions:
n = size(A,2);  % state
m = size(B,2);  % input
q = size(C,1);  % output

%% M-matrix:
% For neater coding:
M = Der - kappa*(Der-Atilde).'*Piw*(Der-Atilde) - kappa*Ctilde.'*Piz*Ctilde;

%% A-matrix:
% The state of the model is [x mu u], hence the 3x3 partitioning of Acl:
Acl = [A                     zeros(n,n*(p+1))    B;...
       kappa*Ctilde.'*Piz*C  M                   zeros(n*(p+1),m);...
      -rho*G*Piz*C        rho*G*Piz*Ctilde zeros(m,m)];

%% B-matrix:
% The inputs are [xi w z], hence the 3x3 partitioning of Bcl: 
Bcl = [zeros(n,n*(p+1))            eye(n)            zeros(n,q);...
       kappa*(Der-Atilde).'*Piw    zeros(n*(p+1),n)  kappa*Ctilde.'*Piz;... 
       zeros(m,n*(p+1))            zeros(m,n)       -rho*G*Piz];

%% C-matrix:
% For inspection purposes, we choose to measure all signals:
Ccl = eye(size(Acl));

%% D-matrix:
% There is no direct feedthrough of any signal:
Dcl = 0;

%% Return state space model of closed loop:
% Put the four matrices in a state space sturcture (continuous):
AIss = ss(Acl,Bcl,Ccl,Dcl);

end

