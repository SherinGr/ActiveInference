function F = freeEnergy(mu,y,agent)
%% Evaluate current Informational Free Energy (IFE)
% Extract required variables 
Atilde = agent.Atilde;
Ctilde = agent.Ctilde;
D      = agent.Der;
Piz    = agent.Piz;
Piw    = agent.Piw;
x_eq   = agent.x_eq;
p      = agent.p;

% Calculate the Free Energy
N = size(y,2); % number of timepoints
F = NaN(1,N);  % initialize vector

n = numel(x_eq);
xi = -Atilde*[x_eq; zeros(n*p,1)];

for i = 1:N
F(i) = 0.5*((y(:,i)-Ctilde*mu(:,i)).'*Piz*(y(:,i)-Ctilde*mu(:,i)) + ...
    (D*mu(:,i)-Atilde*mu(:,i)-xi).'*Piw*(D*mu(:,i)-Atilde*mu(:,i)-xi));
end

end

