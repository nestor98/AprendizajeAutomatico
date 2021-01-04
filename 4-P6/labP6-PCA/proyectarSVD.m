function [Xhat, Xhat2] = proyectarSVD(U, S, V, k)
% Devuelve en Xhat la reconstrucci�n de los datos originales con las
% primeras k sumas parciales de los vectores en U, S y V seg�n el m�todo
% SVD
Xhat = 0; % empieza en 0
for i = 1:k
    % Sumatorio de 1 a k de las siguientes multiplicaciones de los
    % vectores (V se debe trasponer)
    Xhat = Xhat + U(:,i) * S(i,i) * V(:,i)';
    
end
Xhat2 = (U(:,1:k) * S(1:k,1:k) * V(:, 1:k)');


