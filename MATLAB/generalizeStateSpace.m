function [Atilde,Btilde,Ctilde,Dtilde] = generalizeStateSpace(process,p,genmeas)
%% Construct a generalized state space model
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     25-09-2018
%
% INPUTS:
% process   =   State space model of generative process. Can be either 
%               continuous or discrete time model.
% p         =   Embedding order (>=1)
% genmeas   =   Optional boolean, indicating availability of generalized
%               measurements.

% OUTPUTS:
% Atilde    =   A matrix of generatized model
% Btilde    =   B matrix of generalized model
% Ctilde    =   C matrix of generalized model
% Dtilde    =   D matrix of generalized model

% NOTE:
% This function generalizes all four matrices, for generality. However, 
% input is not modeled in the generative model for active inference, 
% so only Atilde and Ctilde are required in active inference applications.

%% Pre-processing:
A = process.A;
B = process.B;
C = process.C;
D = process.D;

n = size(A,2); % state dimension
m = size(B,2); % input dimension
q = size(C,1); % output dimension

% Check whether user defined preference for generalized measurement:
switch nargin
    case 2
        generalizeMeasurements = false; % default = NO generalized measurement
    case 3
        generalizeMeasurements = genmeas;
end

%% Generalize the model:
% p = 1 -> copy original model. 
% p = 2 -> adds one extra motion, etc. etc.

Ip = eye(p);

Atilde = kron(Ip,A);
Btilde = kron(Ip,B);

% Two options for C and D, depending on assumptions:
switch generalizeMeasurements
    case false % NO generalized measurement available
        Ctilde = [C zeros(q,n*(p-1))]; 
        Dtilde = [D zeros(q,m*(p-1))];
    case true % generalized measurement available
        Ctilde = kron(Ip,C);         
        Dtilde = kron(Ip,D);
end

