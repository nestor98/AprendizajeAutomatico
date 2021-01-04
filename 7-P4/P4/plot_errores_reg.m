function [] = plot_errores_reg(valores_err_T, valores_err_V, eje_x, label_eje_x, label_eje_y, titulo)
%PLOT_ERRORES Summary of this function goes here
%   Detailed explanation goes here
figure;
semilogx(eje_x, (valores_err_T), 'b');
grid on; hold on;
semilogx((eje_x), (valores_err_V), 'r');
title(sprintf(titulo));
ylabel(label_eje_y); xlabel(label_eje_x);
% Xd = expandir(x1dibu, grado);
% plot(x1dibu, Xd*w, 'r-'); % Dibujo la recta de predicción
legend('Errores Train', 'Errores Validacion');%, 'Prediccion')

end
