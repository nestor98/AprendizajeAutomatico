function [J,grad,Hess] = costeL2(theta,X,y)% Calcula el coste cuadrático, y si se piden,  
    % su gradiente y su Hessiano 
    r = X*theta-y;
    J = (1/2)*sum(r.^2);
    if nargout > 1  
        grad = X'*r; 
    end
    if nargout > 2  
        Hess = X'*X; 
    end
end

