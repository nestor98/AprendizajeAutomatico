function [mu_res] = residuo_hab(datos, pred_y, habitaciones)
%RESIDUO_HAB Summary of this function goes here
%   Detailed explanation goes here

prom = mean(datos(datos(:,1)==100 & datos(:,2)==habitaciones, 3));
mu_res = abs(prom - pred_y(habitaciones-1));
end

