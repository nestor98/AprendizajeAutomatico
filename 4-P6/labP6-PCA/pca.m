function [U, Lambda] = pca(X)
% Asigna a U los vectores propios de la matriz de covarianzas de X
% y los ordena en función de sus valores propios, asignados a Lambda
sigma = matCov(X);
[U, Lambda] = eig(sigma);
% Ordenar los vectores y valores propios de mayor a menor valor propio
% de la doc de Matlab (https://es.mathworks.com/help/matlab/ref/eig.html)
[vals,ind] = sort(diag(Lambda),'descend');

Lambda = Lambda(ind,ind); % Ordenamos lambda en f de los indices (descend = en orden descendente)
% reordenar U:
U = U(:,ind);

end

