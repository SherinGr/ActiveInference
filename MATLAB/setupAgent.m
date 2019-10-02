function agent = setupAgent(process,p,kappa,rho,varw,varz,s)
%% Construct a structure containing the generative model variables
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     25-09-2018
%
% INPUTS:
% process   =   State space model of generative process (assuming agent
%               knows the exact process dynamics). Can be either continuous
%               or discrete time model.
% p         =   Desired number or generalized coordinates (>=0)
% kappa     =   Learning rate for perception cycle (>0)
% rho       =   Learning rate for action cycle (>0)
% varw      =   Variance of the state noises 
% varz      =   Variance of the output noises
% s         =   Smoothness of the noises (same for all)
%
% OUTPUTS:
% agent.A      =   A matrix of generative model
% agent.B      =   B matrix of generative model
% agent.C      =   C matrix of generative model
% agent.G      =   Forward Model of agent
% agent.D      =   Derivative operator matrix
% agent.p      =   Embedding order
% agent.Piz    =   Precision matrix output noise
% agent.Piw    =   Precision matrix state noise
% agent.kappa  =   Perception learning rate
% agent.rho    =   Action learning rate
% agent.s      =   Smoothness of the noises

%% Agent model options:
% Choose whether or not to use generalized measurements:
generalizeMeasurements = false;
% Note: this is hard-coded to false because the functionality for TRUE is not 
% there yet. 

%% Pre-processing:
% Extract process matrices:
A = process.A;
B = process.B;
C = process.C;

% Dimension variables:
n = size(A,1); % state dimension

% Identity matrices for quick construction:
Ip = eye(p);
In = eye(n);

%% Model Construction:
% Set up generative model (assuming process is known):
Ahat = A;
Bhat = B;
Chat = C;
model = ss(Ahat,Bhat,Chat,[]);

% Generalize the model:
[Atilde,~,Ctilde,~] = generalizeStateSpace(model,p,generalizeMeasurements);

% Construct forward model:
Ghat = forwardModel(model);

% Construct D-matrix (derivative operator):
if p<1 
    warning('p cannot be smaller than 1!');
elseif p==1
    D = 0; % WRONG!! This is a special case, does not work with this code yet.
else 
    T = toeplitz(zeros(1,p),[0 1 zeros(1,p-2)]);
    D = kron(T,In);
end

%% Set up generalized precision (inverse covariance) matrices:
if ~generalizeMeasurements
    Piz_tilde = diag(1./varz);
else 
    Piz_tilde = kron(Ip,diag(1./varz)); % not used yet
end

S = temporalVarianceMatrix(p,s);
Piw = diag(1./varw);
Piw_tilde = kron(S,Piw);

%% Generate the output structure:
agent.A = Atilde;
agent.C = Ctilde;
agent.G = Ghat;
agent.B = Bhat;
agent.D = D;
agent.p = p;
agent.Piz = Piz_tilde;
agent.Piw = Piw_tilde;
agent.kappa = kappa;
agent.rho = rho;
agent.s = s;

agent.ygen = generalizeMeasurements; % Temporal, to catch problems later on.

end

