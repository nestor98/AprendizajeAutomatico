function [yhat] = clasificar_multi(X, theta)
% clasifica en yhat las muestras X en funcion de theta y el umbral
% (cada columna de theta son los pesos de una clase)

probabilidades = []; % probabilidad de cada clase
for i = 1:size(theta, 2) % para cada columna de theta (clase)
    [~, prob]= clasificar_binaria(X, theta(:,i));
    probabilidades = [probabilidades prob];
end

% el segundo argout es el indice, que se corresponde con la clase de max
% prob:
[~, yhat] = max(probabilidades, [], 2); 
% el ultimo argin (2) es para obtener el max de cada fila (cada fila es una prob)

end

