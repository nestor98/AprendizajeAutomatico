function [k, varK] = obtenerK(vals, varMin)
% asigna a k el menor valor tal que se mantiene el varMin de la
% variabilidad de los datos. vals es un vector vertical con los valores
% propios (o singulares) ordenados de mayor a menor
varTot = sum(vals);
for i = 1:size(vals) % para cada valor de k en [1,3]
    % calculamos su var relativa
    varK = sum(vals(1:i))/varTot;
    if varK >= varMin % Y si es mayor que la mínima buscada, terminamos
        k = i; % Asignando a k su valor
        break;
    end
end

end

