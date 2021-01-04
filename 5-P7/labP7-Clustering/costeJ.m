function [J] = costeJ(D, mu, c)
% devuelve en J el valor de coste L2 de la asignacion c a los datos D con
% los centroides con valores mu
r = (D- mu(c,:)).^2; % distancia al cuadrado de cada punto real a su media asignada
J = mean(mean((r),2)); % J = suma de las distancias
% sum(r,2) para que sume cada fila en vez de cada columna

end

