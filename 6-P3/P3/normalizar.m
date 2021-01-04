function [ Xn, mu, sd ] = normalizar( X, mu, sd )
% Normaliza los atributos
% Si se invoca con mu y sd, se hace a partir de estos valores, 
% si no, se calculan para X
if nargin < 2
    mu = mean(X(:,2:end));
end
if nargin < 3
    sd = std(X(:,2:end));
end
Xn = X;
% N = size(X,1);
% Xn(:,2:end) = ( X(:,2:end) - repmat(mu,N,1) )./ repmat(sig,N,1);
for i = 2:size(X,2)
    Xn(:,i) = (X(:,i) - mu(i-1)) / sd(i-1);
end

end

