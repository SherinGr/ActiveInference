function [xi,ref] = dynamicPrior(Atilde,D,p,t)
%% Construct a dynamic prior, for reference tracking
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     06-01-2019
%
% INPUTS:
% Atilde    =   Belief of the agent about the generative process
% D         =   The derivative operator matrix
% ref       =   Desired reference state for generative process
% p         =   Number of generalized states
% t         =   Time vector of simulation
%
% OUTPUTS:
% xi        =   Prior  input that will steer the estimation dynamics
%               for the whole duration of the simulation
% ref       =   The standard reference function
%
% NOTE:
% At this point the reference input is hard-coded, this can be improved in
% a future version by using it as input to this function.

%% Pre-processing:
n  = size(Atilde,1)/p;
N  = numel(t);

%% Hard-coded reference (sinusoid):
syms x
% Sinusoid:
ref = 40*sin(x/1.5);

% Find its generalized form with symbolic differentiation:
xi = [];
for i = 0:p-1
    xi = [xi;diff(ref,i)];
end
% Subsitute and evaluate for the time vector:
x = t;
ref = eval(xi);

% Chop of the first part and replace with zeros:
ref = zeros(n*p,round(N/6)); % first part zeros
x   = t(round(N/6)+1:end);
ref = [ref eval(xi)];        % add sinusoid

%% Construct the prior signal for simulation:
xi = (D-Atilde)*ref;      % static prior variable

end