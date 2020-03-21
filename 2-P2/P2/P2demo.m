close all;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Años, Km, CV
x1dibu = linspace(min(Xdatos(:,1)), max(Xdatos(:,1)), 100)'; %para dibujar

datos2 = load('CochesTest.txt');
ytest = datos2(:,1);  % Precio en Euros
Xtest = datos2(:,2:4); % Años, Km, CV
Ntest = length(ytest);


%% Dibujo de un Ajuste Parabólico Monovariable

disp('********************Ajuste polinomico multivariable*******************');
grado = 9;
X = expandir(Xdatos, [grado 0 0]);
[ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, X, ydatos );
w = Xtr\ytr;  % Solucion con la Ecuación Normal

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



%% 2. Grado pol antiguedad

disp('********************2. Grado pol antiguedad*******************');
% function [hypothesis, mejor_grado, h_mu, h_sd] = kfold_cv_grados(Learner, k, X, y, i, grados)
grados = [10 1 1]; % probar grados [1..10 1 1]
indice = 1; % indice del parametro a optimizar (el primero en este caso)
k = 5; % numero de folds
mejor_grado=0;
mu = 0;sd = 0;
[w_a, mejor_grado_a, mu, sd] = kfold_cv_grados(k,Xdatos,ydatos,indice,grados);
% grado = 9;
% X = expandir(Xdatos, [grado 0 0]);
% [ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, X, ydatos );
% w = Xtr\ytr;  % Solucion con la Ecuación Normal
Xtest_exp = expandir(Xtest, [mejor_grado_a 1 1]);
% [Xtest_exp_n, mu, sd] = normalizar(Xtest_exp);
mejor_grado_a
RMSE_test_a = RMSE( w_a, Xtest_exp, ytest)


%% 3. Grado pol km

disp('********************2. Grado pol kilometros*******************');
% function [hypothesis, mejor_grado, h_mu, h_sd] = kfold_cv_grados(Learner, k, X, y, i, grados)
grados = [mejor_grado_a 10 1]; % probar grados [1..10 1 1]
indice = 2; % indice del parametro a optimizar (el segundo en este caso)
k = 5; % numero de folds
mejor_grado_km=0;
mu = 0;sd = 0;
[w_km, mejor_grado_km, mu, sd] = kfold_cv_grados(k,Xdatos,ydatos,indice,grados);
% grado = 9;
% X = expandir(Xdatos, [grado 0 0]);
% [ Xcv, ycv, Xtr, ytr ] = particion( 1, 5, X, ydatos );
% w = Xtr\ytr;  % Solucion con la Ecuación Normal
Xtest_exp = expandir(Xtest, [mejor_grado_a mejor_grado_km 1]);
% [Xtest_exp_n, mu, sd] = normalizar(Xtest_exp);
mejor_grado_km
RMSE_test_km = RMSE( w_km, Xtest_exp, ytest)

%% 4. Regularizacion

disp('********************4. Regularizacion*******************');
% function [hypothesis, mejor_grado, h_mu, h_sd] = kfold_cv_grados(Learner, k, X, y, i, grados)
grados = [10 5 5]; % probar grados [1..10 1 1]
%indice = 2; % indice del parametro a optimizar (el segundo en este caso)
k = 5; % numero de folds
mejor_grado_km=0;
% potencias = -6:0;
probar_lambdas = logspace(-20, 0, 500);%10.^potencias % [10^-6 10^-5 ... 10^6]

[w_reg, mejor_lambda, mu, sd] = kfold_cv_reg(k,Xdatos,ydatos,grados, probar_lambdas);

Xtest_exp = expandir(Xtest, grados);
% [Xtest_exp_n, mu, sd] = normalizar(Xtest_exp);
mejor_lambda
RMSE_test_reg = RMSE( w_reg, Xtest_exp, ytest)