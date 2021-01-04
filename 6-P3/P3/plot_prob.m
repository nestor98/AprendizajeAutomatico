function plot_prob(X, y, titulo, s_ylabel, s_xlabel);
% Dibuja una grafica de y en funcion de X con el titulo y los labels dados
figure;
grid on; hold on;
plot(X, y, 'b');
title(sprintf(titulo));
ylabel('J'); xlabel(s_xlabel); ylabel(s_ylabel);
end

