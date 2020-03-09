close all;
%% Cargar los datos
datos = load('PisosTrain.txt');
test = load('PisosTest.txt');
y = datos(:,3);  % Precio en Euros
x1 = datos(:,1); % m^2
x2 = datos(:,2); % Habitaciones
N = length(y);

y_test = test(:,3);  % Precio en Euros
x1_test = test(:,1); % m^2
x2_test = test(:,2); % Habitaciones
N_test = length(y_test);
%% 2. Ecuacion normal, monovariable
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

X_mono = [ones(N,1) x1];
th_mono = X_mono\y
%th = [5000 1000]';  % Pongo un valor cualquiera de pesos
Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * th_mono

plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción


str_prediccion_ec_mono = strcat('Prediccion: precio = ', mat2str(th_mono(2)), '*S(m^2) + ', mat2str(th_mono(1)))

legend('Datos Entrenamiento', str_prediccion_ec_mono);

%% 2- residuos mono train, test
residuos_train = RMSE(th_mono, X_mono, y);
X_mono_test = [ones(N_test,1) x1_test];
residuos_test = RMSE(th_mono, X_mono_test, y_test);
residuos = [(residuos_train) (residuos_test)]

%% 3a. Eq. normal, multivariable

X_multi = [ones(N,1) x1 x2];
th_multi = X_multi\y;  % Se resuelve la ecuación
yest = X_multi * th_multi;

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th_multi;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos');
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
%str_ecuacion = strcat('h(x) = ', mat2str(th), '*x (€)');
strcat('h(x) = ', mat2str(th_multi), '*[1, S(m^2), num_hab] (€)')

%% 3a.- residuos
residuos_train = y-yest;
mu_res_abs_train = mean(abs(residuos_train))

X_multi_test = [ones(N_test,1) x1_test x2_test];
yest_test = X_multi_test * th_multi;
residuos_test = y_test-yest_test;
mu_res_abs_test = mean(abs(residuos_test))


rmse_train_multi = RMSE(th_multi, X_multi, y);
rmse_test_multi = RMSE(th_multi, X_multi_test, y_test);
rmse_multi = [rmse_train_multi rmse_test_multi]
% Dibujar los puntos de entrenamiento y su valor estimado 
%figure;  
%plot3(x1, x2, residuos, '.r', 'markersize', 20);
%axis vis3d; hold on;
%plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

%% 3b.- comparacion 3a
superficie = 100;
ysup_test = y_test(find(x1_test==100)); % precios reales de los pisos en los datos de test
% con monovariable:
Xsup = [1 superficie];
ysup_mono = Xsup * th_mono
prom_real_100 = mean(ysup_test)

residuo_mono = repmat(ysup_mono,size(ysup_test)) - ysup_test;
mu_res_abs_mono = mean(abs(residuo_mono))

rmse_mono_100 = RMSE(th_mono, Xsup, y(x1==100))
rmse_mono_test_100 = RMSE(th_mono, Xsup, y_test(x1_test==100))
% con multi:
habs = [2 3 4 5]';
Xsup = [ones(size(habs)) repmat(superficie, size(habs)) habs];
ysup_multi = Xsup * th_multi

prom_real_100_hab = mean(y_test(x1_test==100 & x2_test==5)) %& x2_test==habs))

%rmse de los ajustes multi con 100 metros 
% rmse_multi_100 = RMSE(th_multi, Xsup, y(x1==Xsup(2) && x2==Xsup(3)))
% rmse_multi_test_100 = RMSE(th_multi, Xsup, y_test(x1_test==100))

%Residuo medio (absoluto) con 2 a 5 hab:
res = [];
for hab = 2:5
    res(hab) = residuo_hab(datos, ysup_multi, hab);
end
mean(res(2:4))

%% 4. Descenso de gradiente, monovariable
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on; 

% Normalizacion:
% N = size(X,1);
% mu = mean(X(:,2:end));
% sd = std(X(:,2:end));
% X(:,2:end) = (X(:,2:end) - repmat(mu,N,1)) ./ repmat(sd, N, 1)
% tb podria haber llamado a la funcion normalizar...


X_mono = [ones(N,1) x1]; 
M = 200000; % numero de iteraciones máximas
alfa = [0.000000001; 0.000000005; 0.00000001; 0.00000005; 0.0000001; 
    0.000000482; 0.00000048535; 0.000000485;  0.000001; 0.001; 0.01; 0.1]; % distintos valores de alfa a probar
th_g_mono = randn(2, 1); % N numeros aleatorios con distrib normal
J_iter = [];
for k = 1:M
    [J,g] = costeL2(th_g_mono, X_mono, y);
    if k>2 && J > J_iter(length(J_iter)) % si ha aumentado el error, terminamos 
        break
    end
    J_iter = [ J_iter; J ];
    th_g_mono = th_g_mono - alfa(7) .* g;
end


Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * th_g_mono;  
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Prediccion');

str_prediccion = strcat('Prediccion: precio = ', mat2str(th_g_mono(2)), '*S(m^2) + ', mat2str(th_g_mono(1)))
legend('Datos Entrenamiento', str_prediccion);
strcat('con gradiente: ', str_prediccion)
strcat('con ecuacion normal: ', str_prediccion_ec_mono)
% Son distintos porque la ecuación normal va en este caso al mínimo global
% directamente, mientras que el gradiente se aproxima iterativamente

[th_mono, th_g_mono]

% Grafica completa del coste: 
figure;
plot(1:M, log(J_iter), 'bx');
title('Coste por iteracion');
ylabel('log Coste'); xlabel('Iteracion');
grid on; hold on;

% Segunda mitad (para reducir la escala):
% figure;
% plot(M/2:M, log(J_iter(M/2:end)), 'bx');
% title('Coste por iteracion')
% ylabel('Coste'); xlabel('Iteracion');
% grid on; hold on;




% X = [ones(N,1) x1];
% th_mono = X\y
% %th = [5000 1000]';  % Pongo un valor cualquiera de pesos
% Xextr = [1 min(x1)  % Predicción para los valores extremos
%          1 max(x1)];
% yextr = Xextr * th_mono
% 
% plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
% 
% 
% str_prediccion = strcat('Prediccion: precio = ', mat2str(th_mono(2)), '*S(m^2) + ', mat2str(th_mono(1)))
% 
% legend('Datos Entrenamiento', str_prediccion);


%% 5. Descenso de gradiente, multivariable

% Normalizacion:
X = [ones(N,1) x1 x2];
[Xn, mu, sd] = normalizar(X);
% N = size(X,1);
% mu = mean(X(:,2:end));
% sd = std(X(:,2:end));
% X(:,2:end) = (X(:,2:end) - repmat(mu,N,1)) ./ repmat(sd, N, 1)
% tb podria haber llamado a la funcion normalizar...


M = 1000; % numero de iteraciones
alfa = [0.000000001; 0.000000005; 0.00000001; 0.00000005; 0.0000001; 
    0.000000482; 0.0000004853; 0.000000485;  0.000001; 0.001; 0.01; 0.1]; % distintos valores de alfa a probar
th_g_multi = randn(3, 1); % N numeros aleatorios con distrib normal
J_iter = [];
for k = 1:M
    [J,g] = costeL2(th_g_multi, Xn, y);
    J_iter = [ J_iter; J ];
    th_g_multi = th_g_multi - alfa(10) .* g;
end


% str_prediccion = strcat('Prediccion: precio = ', mat2str(th_g_mono(2)), '*S(m^2) + ', mat2str(th_g_mono(1)))
% legend('Datos Entrenamiento', str_prediccion);
% strcat('con gradiente: ', str_prediccion)
% strcat('con ecuacion normal: ', str_prediccion_ec_mono)
% Son distintos porque la ecuación normal va en este caso al mínimo global
% directamente, mientras que el gradiente se aproxima iterativamente


% Grafica completa del coste: 
figure;
plot(1:M, log(J_iter), 'bx');
title('Coste por iteracion');
ylabel('log Coste'); xlabel('Iteracion');
grid on; hold on;

% x1n y x2n, normalizados:
% x1n = Xn(:,1); % m^2
% x2n = Xn(:,2); % Habitaciones

% Desnormalizar pesos:
th_g_multi = desnormalizar(th_g_multi, mu, sd);

ths = [th_g_multi th_multi]

% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th_g_multi;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos');
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');


%% 5. Comparacion 3,4,5
superficie = 100;
ysup_test = y_test(find(x1_test==100)); % precios reales de los pisos en los datos de test

% -------------------------- Ecuacion normal:
% con monovariable:
Xsup = [1 superficie];
ysup_ec_mono = Xsup * th_mono;

residuo_mono = repmat(ysup_ec_mono,size(ysup_test)) - ysup_test;
mu_res_abs_mono = mean(abs(residuo_mono))

% con multi:
habs = [2 3 4 5]';
Xsup = [ones(size(habs)) repmat(superficie, size(habs)) habs];
ysup_ec_multi = Xsup * th_multi;

% Residuo medio (absoluto) con 2 a 5 hab:
res = [];
for hab = 2:5
    res(hab) = residuo_hab(datos, ysup_ec_multi, hab);
end
mean(res(2:4))



% -------------------------- Descenso de gradiente:
% con monovariable:

Xsup = [1 superficie];
ysup_g_mono = Xsup * th_g_mono;

residuo_g_mono = repmat(ysup_g_mono,size(ysup_test)) - ysup_test;
mu_res_abs_g_mono = mean(abs(residuo_g_mono))



% con multi:
habs = [2 3 4 5]';
Xsup = [ones(size(habs)) repmat(superficie, size(habs)) habs];
ysup_g_multi = Xsup * th_g_multi;

% Residuo medio (absoluto) con 2 a 5 hab:
res = [];
for hab = 2:5
    res(hab) = residuo_hab(datos, ysup_g_multi, hab);
end
mean(res(2:4))

rmse = [ RMSE(th_mono, X_mono, y) RMSE(th_multi, X_multi, y);
         RMSE(th_g_mono, X_mono, y) RMSE(th_g_multi, X_multi, y)]
     
     
     
     
%% 6. Descenso de gradiente, coste de Huber

% Normalizacion:
X = [ones(N,1) x1 x2];
[Xn, mu, sd] = normalizar(X);
% N = size(X,1);
% mu = mean(X(:,2:end));
% sd = std(X(:,2:end));
% X(:,2:end) = (X(:,2:end) - repmat(mu,N,1)) ./ repmat(sd, N, 1)
% tb podria haber llamado a la funcion normalizar...


M = 100000; % numero de iteraciones
alfa = [0.000000001; 0.000000005; 0.00000001; 0.00000005; 0.0000001; 
    0.000000482; 0.0000004853; 0.000000485;  0.000001; 0.001; 0.01; 0.1; 1; 10; 300; 4]; % distintos valores de alfa a probar
th_g_multi_h = randn(3, 1); % N numeros aleatorios con distrib normal
J_iter = [];
d = 0.0001; % delta para el coste de Huber
for k = 1:M
    [J,g] = costehuber(th_g_multi_h, Xn, y, d);
    J_iter = [ J_iter; J ];
    th_g_multi_h = th_g_multi_h - 5000.* g;
end


% str_prediccion = strcat('Prediccion: precio = ', mat2str(th_g_mono(2)), '*S(m^2) + ', mat2str(th_g_mono(1)))
% legend('Datos Entrenamiento', str_prediccion);
% strcat('con gradiente: ', str_prediccion)
% strcat('con ecuacion normal: ', str_prediccion_ec_mono)
% Son distintos porque la ecuación normal va en este caso al mínimo global
% directamente, mientras que el gradiente se aproxima iterativamente


% Grafica completa del coste: 
figure;
plot(1:M, log(J_iter), 'bx');
title('Coste por iteracion');
ylabel('log Coste'); xlabel('Iteracion');
grid on; hold on;

% x1n y x2n, normalizados:
% x1n = Xn(:,1); % m^2
% x2n = Xn(:,2); % Habitaciones

% Desnormalizar pesos:
th_g_multi_h = desnormalizar(th_g_multi_h, mu, sd);



% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');

% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);

% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * th_g_multi_h;

% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos');
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');


%% Comparar todos:

rmse = [ RMSE(th_mono, X_mono, y) RMSE(th_multi, X_multi, y);
         RMSE(th_g_mono, X_mono, y) RMSE(th_g_multi, X_multi, y);
         RMSE(th_g_mono, X_mono, y) RMSE(th_g_multi_h, X_test, y);
         ]
     
     


