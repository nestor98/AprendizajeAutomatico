function [ Xn ] = normalizar_mu_sd( X, mu, sd )
% Normaliza los atributos de X en funcion de mu y sd

Xn = X;
% N = size(X,1);
% Xn(:,2:end) = ( X(:,2:end) - repmat(mu,N,1) )./ repmat(sig,N,1);
for i = 2:size(X,2)
    Xn(:,i) = (X(:,i) - mu(i-1)) / sd(i-1);
end

end

