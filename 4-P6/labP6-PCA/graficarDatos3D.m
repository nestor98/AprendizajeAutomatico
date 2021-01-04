function [] = graficarDatos3D(X, nFig)
% GRAFICARDATOSXY Gráfico de dispersión de las tres primeras dimensiones de
% X (que se rotularán como X, Y, Z). 
% nFig simplemente determina el número con el que identificar la figura.
figure(nFig);
axis equal;
grid on;
hold on;
plot3(X(:,1),X(:,2),X(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
pause
end

