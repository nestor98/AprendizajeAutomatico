function plotJ(J, titulo, str_xlabel)
% Dibuja una grafica de J en funcion de la iteracion con el titulo dado
figure;
grid on; hold on;
plot(1:size(J,1)', J, 'b');
title(sprintf(titulo));
ylabel('J'); xlabel(str_xlabel);
% Xd = expandir(x1dibu, grado);
% plot(x1dibu, Xd*w, 'r-'); % Dibujo la recta de predicción
% legend('Errores Train', 'Errores Validacion');%, 'Prediccion')

end

