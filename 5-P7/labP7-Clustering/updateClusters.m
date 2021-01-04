function Z = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class

K = size(mu,1); % numero de centroides (clases)

% Z asigna 1..K a cada fila de D

distancias = []; % almacenara las distancias de cada pixel a cada centroide

% en cada fila estaran las distancias a cada clase del pixel de esa
% fila en D

for fila = D'
   dist = mean((fila' - mu).^2, 2); % distancia del pixel <fila> a cada centroide
   distancias = [distancias dist]; % actualizamos la matriz
end

[~,Z] = min(distancias); % Z array con la asignacion de cada pixel a una clase
% el segundo arg almacena los indices (no nos interesan los valores)

Z = Z'; % para que queden en una columna como D