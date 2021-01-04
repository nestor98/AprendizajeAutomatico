function [n] = nMuestrasClase(ytr, clase)
% n = numero de muestras de la clase <clase> en ytr
n = sum(ytr == clase); % num de eltos = <clase>    (:)
end

