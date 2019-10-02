function S = temporalVarianceMatrix(p,s)
%% Returns the inverse temporal variance matrix of the generalised noises
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     06-11-2018
%
% Note: This file is largely inspired by spm_DEM_R.m by Karl Friston.
%
% INPUTS:
% p             = number of generalised states (>=1)
% s             = smoothness of the noise
%
% OUTPUTS:
% S             = inverse temporal covariance matrix for generalised noises

k          = 0:(p - 1);
r(1 + 2*k) = cumprod(1 - 2*k)./(s.^(2*k));

% Covariance matrix:
V     = [];
for i = 1:p
    V = [V; r([1:p] + i - 1)];
    r = -r;
end

% Smoothness matrix:
S = inv(V);

