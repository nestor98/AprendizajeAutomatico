%% Practica 6.1: PCA 

clear all
close all

%% Leer los datos originales en la variable X
load P61

% Graficar los datos originales
graficarDatos3D(X, 1)

%% Estandarizar los datos (solo hace falta centrarlos)
[Xn, mu] = estandarizar(X);

% Graficar los datos centrados
graficarDatos3D(Xn, 2);

%% Calcular la matrix de covarianza muestral de los datos centrados
sigma = matCov(Xn)

%% Aplicar PCA para obtener los vectores propios y valores propios
[U, Lambda] = eig(sigma)

%% Ordenar los vectores y valores propios de mayor a menor valor propio
% de la doc de Matlab (https://es.mathworks.com/help/matlab/ref/eig.html)
[vals,ind] = sort(diag(Lambda),'descend')

Lambda = Lambda(ind,ind) % Ordenamos lambda en f de los indices (descend = en orden descendente)
% reordenar U:
U = U(:,ind)


%% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio

%los 3 vectores:
dibujarVect(U, Lambda);

% vectores propios, los dos primeros, o solo el primer vector propio
% Graficar la variabilidad que se mantiene si utilizas los tres primeros

%% Variabilidades parar cada valor de k:
disp('var total:');
varTot = sum(vals)

varK = [];

for k = 1:size(vals) % para cada valor de k en [1,3]
    % su var es la suma de la anterior con la actual
    varK = [varK sum(vals(1:k))]; 
end

disp('var absoluta para cada k:');
varK

disp('var relativa para cada k:');
varKRel = varK./varTot


%% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad

disp('Para mantener el 90% de la variabilidad, el menor valor de k es:')
k = obtenerK(vals, 0.9)

% U reducida (Uk, con los k primeros vectores, columnas):
Uk = U(:,1:k)
Z = Xn*Uk;

%% Graficar aparte los datos z proyectados según el resultado anterior


figure();
title('Grafico de los datos Z');
axis equal;
grid on;
hold on;
plot(Z(:,1),Z(:,2),'r.');
xlabel('Eje X');
ylabel('Eje Y');

%% Graficar en verde los datos reproyectados \hat{x} en la figura original
Xhat = (Uk * Z')'

figure();
title('Reproyección junto con los datos originales');
axis equal;
grid on;
hold on;
plot3(Xn(:,1),Xn(:,2),Xn(:,3),'b.'); % originales en azul
plot3(Xhat(:,1),Xhat(:,2),Xhat(:,3),'g.'); % reproyeccion en verde
legend('Datos originales', 'Xhat')
xlabel('Eje X');
ylabel('Eje Y');
zlabel('Eje Z');

% Usando 2 vectores se proyectan los datos en 2D, es decir, un plano
% Cambiando k manualmente se pueden observar las proyecciones a 1D (1 vector)
% o 3D (datos originales)
