% load images 
% images size is 20x20. 
clear
close all

load('MNISTdata2.mat'); 
% Permutamos aleatoriamente los datos:
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);



nrows=20;
ncols=20;

nimages = size(X,1);

% Show the images
for I=1:100:nimages 
    imshow(reshape(X(I,:),nrows,ncols))
    pause(0.1)
end


%% Perform PCA over all numbers


%% Obtener Z
k = 2;
% Estandarizamos los datos:
[Xn, mu] = estandarizar(X);
% Obtenemos vect y val propios ordenados:
[U,Lambda] = pca(Xn);
% Obtenemos Z a partir de lo anterior:
[Xhat, Z] = proyectar_pca(Xn, U, k);

%% Mostrar componentes
% Muestra las dos componentes principales
figure(100)
clf, hold on
plotwithcolor(Z(:,1:2), y);
% Interpretación: 
% el eje X (componente principal) representa la anchura en píxeles del
% dígito?. Por ej, los 1 están más a la izq (menor envergadura) y los 0 (clase 10)
% a la dcha


%% Use classifier from previous labs on the projected space
Xtest_n = Xtest - mu; % estandarizamos las muestras de test

% Creo que con las 2 componentes principales se pueden diferenciar 1 y 0, 
% pero 2 y 5 no

% Probamos a clasificar todos los datos de test:
%% Entrenamos Bayes no ingenuo en Z:
clases = [1:10]; % numero de clases
probar_lambdas = logspace(-5, 0, 20); % lambdas a probar (20 entre 10^-20 y 10)
ingenuo = 0; % con bayes no ingenuo
folds = 5; % 5 folds para que separe cada vez el 20% para validación
[modelo, best_lambda_no_ingenuo] = kfoldcv_gaussianas(Z(:,1:2), y, clases, ingenuo, folds, probar_lambdas);
% kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas)
best_lambda_no_ingenuo

%% 4. Clasificar 
% se clasifican los datos de test a partir de los modelos aprendidos con
% los datos de entrenamiento
[Xtesthat, Ztest] = proyectar_pca(Xtest_n, U, k);

yhattest = clasificacionBayesiana(modelo, Ztest, clases);

%% 4. Evaluar
% Se evalúan, comparando las predicciones con las clases reales de cada
% dato
verConfusiones(Xtest, ytest, yhattest); % Ejemplos de confusiones
matriz_conf = confusionmat(ytest, yhattest); % Matriz de confusiones
% confusionchart(ytest, yhattest); % Misma matriz, pero visual
% obtenemos la precision y recall para cada digito a partir de la matriz:
[precision, recall] = precision_recall(matriz_conf)






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Con solo las clases 1 y 0:
X_red = X(y==10 | y==1,:);
y_red = y(y==10 | y==1);
Xtest_red = Xtest(ytest==10 | ytest==1,:);
ytest_red = ytest(ytest==10 | ytest==1);

%% Obtener Z
k = 2;
% Estandarizamos los datos:
[Xn, mu] = estandarizar(X_red);
% Obtenemos vect y val propios ordenados:
[U,Lambda] = pca(Xn);
% Obtenemos Z a partir de lo anterior:
[Xhat, Z] = proyectar_pca(Xn, U, k);



%% Entrenamos Bayes no ingenuo en Z:
clases = [1 10]; % clases
probar_lambdas = logspace(-7, 0, 20); % lambdas a probar (20 entre 10^-20 y 10)
ingenuo = 0; % con bayes no ingenuo
folds = 5; % 5 folds para que separe cada vez el 20% para validación
[modelo, best_lambda_no_ingenuo] = kfoldcv_gaussianas(Z(:,1:2), y_red, clases, ingenuo, folds, probar_lambdas);
% kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas)
best_lambda_no_ingenuo

%% 4. Clasificar 
% se clasifican los datos de test a partir de los modelos aprendidos con
% los datos de entrenamiento
[Xtesthat, Ztest] = proyectar_pca(Xtest_red-mu, U, k);
yhattest = clasificacionBayesiana(modelo, Ztest, clases);

%% 4. Evaluar
% Se evalúan, comparando las predicciones con las clases reales de cada
% dato
% verConfusiones(Xtest_red, ytest_red, yhattest); % Ejemplos de confusiones
matriz_conf = confusionmat(ytest_red, yhattest); % Matriz de confusiones
% confusionchart(ytest_red, yhattest); % Misma matriz, pero visual
% obtenemos la precision y recall para cada digito a partir de la matriz:
[precision, recall] = precision_recall(matriz_conf)








