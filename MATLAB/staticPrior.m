function [xi] = staticPrior(Atilde,x_eq,p,t)
%% Construct the static part of a prior
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     25-09-2018
%
% INPUTS:
% Atilde    =   Belief of the agent about the generative process
% x_eq      =   Desired (constant!) equilibrium state of generative process
% p         =   Number of generalized states
% t         =   Time vector of simulation
%
% OUTPUTS:
% xi        =   Reference value that will steer the estimation dynamics
%
% NOTE: 
% In an improved version, this function could be used to also generate the
% loop shaping part of the prior. This would mean to change the Atilde of the
% brain to the desired shape by using the matrix Ades. The input Atilde in 
% this case would be the belief of the agent about the process.

%% Pre-processing:
n = numel(x_eq);

%% Construct the prior signal:
x_gen_eq = [x_eq ;zeros(n*(p-1),1)];   % Equilibrium in generalized coordinates

Ades = 0;                          % Desired process dynamics (not functional yet)

% Static prior variable for complete simulation:
xi = -(Atilde+Ades)*x_gen_eq*ones(size(t));      

end