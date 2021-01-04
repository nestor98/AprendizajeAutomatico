function [J,grad,Hess] = costeLogistico(theta, X, y)
% Calcula el coste logistico, y si se piden, 
% su gradiente y su HessianoN = size(X,2);
% div = 1+exp(-(X*theta)); % divisor
% % limitamos el divisor para evitar NaN en J:
% % (inspirado en uno de los comentarios de
% % https://stackoverflow.com/a/35422981)
% tope = 10^80; % 10^30--3.02
% div(div>tope) = tope; % capado para no tener h = 0 y que J devuelva NaN
% div(div<-tope) = -tope; % capado para no tener h = 0 y que J devuelva NaN

% h = 1./div;
h = 1./(1+exp(-(X*theta)));
N = size(y,1);

J = (-y'*log(h) - (1-y')*log(1-h))/N;
% if isnan(J)
%    disp('yep'); 
% end
    
if nargout > 1
    grad = X'*(h-y)/N;
end
if nargout > 2
    R = diag(h.*(1-h));
    Hess = X'*R*X/N;
end