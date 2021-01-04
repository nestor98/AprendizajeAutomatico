function [tasa] = tasa_errores(yhat, y)
% tasa = numero_errores(yhat, y)/N
% donde numero_errores es la suma de predicciones fallidas (yhat!=y)
num_errores = sum(yhat~=y);
tasa = num_errores/size(yhat, 1);
end

