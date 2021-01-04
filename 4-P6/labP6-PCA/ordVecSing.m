function [Uord] = ordVecSing(U, S)
% Ordena en Uord los vectores singulares de U según sus valores singulares,
% en S


% Ordenar los vectores y valores propios de mayor a menor valor propio
% de la doc de Matlab (https://es.mathworks.com/help/matlab/ref/eig.html)
[vals,ind] = sort(diag(Lambda),'descend');

LambdaOrd = Lambda(ind,ind); % Ordenamos lambda en f de los indices (descend = en orden descendente)
% reordenar U:
Uord = U(:,ind);
end

