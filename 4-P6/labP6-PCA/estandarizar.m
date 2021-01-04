function [Xn, mu] = estandarizar(X)
%ESTANDARIZAR: Xn es la versión estandarizada (centrada en 0) de X, muX la
% media original (de X)

mu = mean(X(:,1:end));
Xn = X;
% N = size(X,1);
% Xn(:,2:end) = ( X(:,2:end) - repmat(mu,N,1) )./ repmat(sig,N,1);
for i = 1:size(X,2)
    Xn(:,i) = (X(:,i) - mu(i)); 
end

end

