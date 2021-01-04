function [yhat] = clasificar(Xn, theta, umbral)
% clasifica en yhat las muestras normalizadas Xn en funcion de theta y
% umbral
if nargin <= 2 % si se invoca sin el umbral, por defecto es 0.5
   umbral = 0.5;  
end
h = 1./(1+exp(-(Xn*theta)));
yhat = double(h > umbral);
end

