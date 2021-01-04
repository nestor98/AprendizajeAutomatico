function [Xhat,Z] = proyectar_pca(X, U, k)
%PCA Summary of this function goes here
%   Detailed explanation goes here


% U reducida (Uk, con los k primeros vectores, columnas):
Uk = U(:,1:k);
Z = X*Uk; % matriz Z
Xhat = (Uk * Z')'; % reproyeccion de los datos originales con k vectores

end

