function [w,z] = makeNoise(varz,varw,s,t)
%% Produces analytical noise on states and measurements
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     27-09-2018
%
% Note: This file is largely inspired by spm_DEM_z.m by Karl Friston.
%
% INPUTS:
% process       =   state space model of the generative process
% varz          =   vector with measurement noise variances
% varw          =   vector with state noise variances
% s             =   smoothness parameter of the noises
% t             =   time vector for simulation
% dt            =   time step of the time vector
%
% OUTPUTS:
% w             =   Noise on the states
% z             =   Noise on the measurements
%
% NOTE:
% Friston multiplies noise w with sampling time dt. Unsure if correct!

%% Pre-processing
s = s + exp(-16); % smoothness of fluctuations (+ machine precision)

n = length(varw);    
q = length(varz); 
N = length(t);
dt = t(2) - t(1);

% Temporal convolution matrix:
T  = toeplitz(exp(-t.^2/(2*s^2)));
K  = T;%diag(1./sqrt(diag(T*T')));   % normalization of convolution


%% Generate measurement noise:
P = diag(1./varz);
if norm(P,1) == 0 % if no precision, use i.i.d. noise
    z  = randn(q,N)*K;
elseif norm(P,1) >= exp(16) % 'infinite' precision, no noise
    z  = sparse(q,N);
else % make noise with a certain smoothness
    z  = spm_sqrtm(inv(P))*randn(q,N)*K;
end
    
%% Generate state noise:
P = diag(1./varw);
if norm(P,1) == 0
    w = randn(n,N)*K;
elseif norm(P,1) >= exp(16)
    w = sparse(n,N);
else
    w = spm_sqrtm(inv(P))*randn(n,N)*K;
end

