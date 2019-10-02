function F = freeEnergy(mu,y,agent)
%% Evaluate the Free Energy
%
% @author : Sherin Grimbergen <s.s.grimbergen@student.tudelft.nl>
% date:     22-06-2018
%
% INPUTS:
% mu        = Generalized state estimation sequence
% y         = Measurement sequence
% agent     = Structure containing all variables of the agent
%
% OUTPUTS:
% F         = Values of the free energy

%% Extract required variables 
Atilde = agent.A;
Ctilde = agent.C;
D      = agent.D;
Piz    = agent.Piz;
Piw    = agent.Piw;
xi     = agent.xi;

%% Calculate the Free Energy
N = max(size(y)); % number of timepoints (ugly, fails when dim(y)>N)
F = NaN(1,N);     % initialize free energy vector

for i = 1:N
F(i) = 0.5*((y(:,i)-Ctilde*mu(:,i)).'*Piz*(y(:,i)-Ctilde*mu(:,i)) + ...
    (D*mu(:,i)-Atilde*mu(:,i)-xi(:,i)).'*Piw*(D*mu(:,i)-Atilde*mu(:,i)-xi(:,i)));
end

end

