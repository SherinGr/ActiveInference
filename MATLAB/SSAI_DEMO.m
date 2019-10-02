%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                   %
%                           State Space Active Inference                            %                                                                                     %
%                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Tuning parameters:
%------------------------------------------------------------------------------------
% Learning rates:
agent.rho     = 4e6;
agent.kappa   = 5;

% Number of generalised states:
agent.p       = 6;

% Noise variances:
varw    = .1;
varz    = .1;

% Noise smoothness:
s       = .1;



%% System definition:
%------------------------------------------------------------------------------------
% One dimensional (LTI) state space
A = -0.8;
B = 1e-3;
C = 1;
D = 0;

% Put it in a state space structure (to use lsim):
process = ss(A,B,C,D);

% Desired equilibrium state:
agent.x_eq = 10;



%% Preprocessing:
%------------------------------------------------------------------------------------
% Time:
T_end = 5;
dt    = 0.01;
t     = 0:dt:T_end;
N     = numel(t);

% Dimensions:
n = size(A,2); % state dimension
m = size(B,2); % input dimension
q = size(C,1); % output dimension
p = agent.p;   % embedding order

% Identity matrices for quick construction:
Ip = eye(p+1);
In = eye(n);

% Derivative operator:
if p == 0
    agent.Der = 0;
else 
    T = toeplitz(zeros(1,p+1),[0 1 zeros(1,p-1)]);
    agent.Der = kron(T,In);
end



%% Active Inference components:
%------------------------------------------------------------------------------------
% Generalised state space matrices:
agent.Atilde = kron(Ip,A);
agent.Ctilde = [C zeros(q,n*p)];
% NOTE: We assume NO generalised measurement is available

% Forward model:
agent.G = -C*(A\B);
% NOTE: This represents dy/du with steady state assumption. 
% (To find dF/du = dF/dy*dy/du)

% Precision matrices:
agent.Piz = diag(1./varz);
agent.Piw = kron(Ip,diag(1./varw));
% NOTE: This should probably involve the temporal smoothness matrix somehow.



%% Construct closed loop model:
%------------------------------------------------------------------------------------
CL = closeLoop(process,agent);



%% Analytic Noise generation
%------------------------------------------------------------------------------------
% Temporal convolution matrix:
T  = toeplitz(exp(-t.^2/(2*s^2)));
K  = diag(1./sqrt(diag(T*T')))*T; 
% Covariance matrices:
Sw = diag(varw);
Sz = diag(varz);
% Generate noises:
w  = spm_sqrtm(Sw)*randn(n,N)*K;
z  = spm_sqrtm(Sz)*randn(q,N)*K;



%% Simulate Active Inference:
%------------------------------------------------------------------------------------
% Reference signal:
ref = -agent.Atilde*[agent.x_eq; zeros(n*p,1)]*ones(1,N);
% NOTE: This is the implementation for static references only

% Generate closed loop input signal:
ucl = [ref;w;z];

% Initial closed loop state [x mu u]:
x0 = [zeros(n,1); zeros(n*(p+1),1); zeros(q,1)];    

% Perform simulation:
result = lsim(CL,ucl,t,x0);  


%% Post-processing:
% Extract states from the results:
x  = result(:,1:n).';
y  = C*x;
mu = result(:,n+1:n*(p+2)).';
u  = result(:,n*(p+2)+1:end).';

% Calculate IFE for whole trajectory:
F = freeEnergy(mu,y,agent);

% Plot the results:
set(0, 'defaulttextinterpreter','LaTex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'DefaultLineLineWidth', 1.2);

figure(1);
subplot(221); hold on; grid on; box on;
plot(t,agent.x_eq*ones(size(t)),'-.');
plot(t,x(1,:),'--'); 
plot(t,mu(1,:));
legend('$x_{eq}$','$x$','$\mu_x$');
title('States');

subplot(222)
plot(t,u); grid on;
title('Control');

subplot(223);
semilogy(t,F); grid on;
xlabel('Time $[s]$');
title('Free Energy');

subplot(224); 
plot(t,w); hold on; grid on;
plot(t,z);
xlabel('Time $[s]$');
title('Noises');

suptitle('State Space Active Inference');

