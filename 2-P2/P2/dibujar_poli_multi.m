function [RMSEtr,RMSEcv] = dibujar_poli_multi(x1dibu, w, Xtr, ytr, Xcv, ycv, grado)
%DIBUJAR_POLI_MULTI Summary of this function goes here
%   Detailed explanation goes here
figure;
grid on; hold on;
plot(Xtr(:,2), ytr, 'bx');
plot(Xcv(:,2), ycv, 'ro');
title(sprintf('Polinomio grado %d', grado));
ylabel('Precio Coches (Euros)'); xlabel('Años');
Xd = expandir(x1dibu, grado);
plot(x1dibu, Xd*w, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Datos validacion', 'Prediccion')

RMSEtr = RMSE(w, Xtr, ytr)
RMSEcv = RMSE(w, Xcv, ycv)

end

