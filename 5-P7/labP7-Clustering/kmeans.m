function [mu, c, Jfin] = kmeans(D, mu0, dibujarJ)
% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

m = size(D, 1);
n = size(D, 2);

J = []; % lista con los costes de cada iteración

% para la primera iteración:
c_ant = 0; % valor inicial de c_ant, cualquier cosa distinta de c
c = 1; % inicial de c, cualquier cosa distinta de c_ant

mu = mu0; % mu inicial, para la primera iteración
while (~isequal(c, c_ant)) % mientras c cambie respecto a la anterior
    c_ant = c; % actualizamos centroides anteriores
    
    % actualizamos los clusters (clase asignada a cada dato (pixel)):
    c = updateClusters(D, mu);
    % y la nueva media de cada grupo de pixeles asignados a cada clase:
    mu = updateCentroids(D, c); 
    if dibujarJ % si dibujamos guardamos todos los Js
        J_actual = costeJ(D, mu, c);
        J = [J; J_actual]; % añadimos el coste actual al array
    end
end
if dibujarJ % si dibujarJ, dibujamos J
    plotJ(J,'Coste J por iteración', 'iteracion');
    Jfin = J(end); % ultimo valor en J
else % si no, calculamos el valor final
    Jfin = costeJ(D, mu, c);
end


   