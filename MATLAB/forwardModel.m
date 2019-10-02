function G = forwardModel(process)
%% Construct the forward model for given state space process
% This is the steady state gain between input and output
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     25-09-2018
%
% INPUTS:
% process   =   state space model (continuous or discrete)
%
% OUTPUTS:
% G         =   forward dynamic model (steady state gain)

%% Unpack the process:
A = process.A;
B = process.B;
C = process.C;
D = process.D;

%% Define forward model (adopting steady state assumption): 
n = size(A,1);

if process.Ts == 0 % continuous system
    G=-C*(A\B)+D;
else % if discrete system
    G=C*((eye(n)-A)\B)+D;
end

end

