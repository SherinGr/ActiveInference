function [y_gen] = generalizeMeasurement(y,p,t,dt)
%% Generate generalized measurement form discrete measurement sequence
% This is done using the embedding operator, which is derived from a Taylor
% expansion, such that y = T*y_gen --> y_gen = inv(T)*y.
%
% Note that this function performs the generalisation for one time-step! It
% can hence be used on-line during a simulation to construct the
% generalized measurements at each point in time.
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     27-09-2018
%
% Note: This file is largely inspired by spm_DEM_embed.m by Karl Friston.
%
% INPUTS:
% y         =   Measurement sequence (must be of size N>=p!)
% p         =   Embedding order (>=1)
% t         =   Current time
% dt        =   Time step of the time vector
%
% OUTPUTS:
% y_gen     =   Generalized sequence of y (containing p derivatives)

%% Pre-processing
[q,N] = size(y);
y_gen = NaN(p*q,1);

% Find which samples from y to consider:
s      = t/dt;
k      = (1:p)  + fix(s - (p + 1)/2);
x      = s - min(k) + 1;
i      = k < 1;
k      = k.*~i + i;
i      = k > N;
k      = k.*~i + i*N;

% Find the transformation matrix:
for i = 1:p
    for j = 1:p
        T(i,j) = ((i - x)*dt)^(j - 1)/prod(1:(j - 1));
    end
end
E     = inv(T);

%% Construct the generalised measurement:
for i = 1:p
    y_gen(1+(i-1):i*q)      = y(:,k)*E(i,:)';
end

