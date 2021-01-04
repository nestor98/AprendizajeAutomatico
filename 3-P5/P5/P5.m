%% Practica 5
% Based on exercise 3 of Machine Learning Online Class by Andrew Ng 
%

clear ; close all;
addpath(genpath('../minfunc'));

%% Cargar datos
% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);

%% Mostrar datos
displayData(X)

%% 3. Entrenar Bayes ingenuo
% Basándote en el código de la práctica anterior, programa el entrenamiento
% y clasificación multi-clase, usando Bayes ingenuo. Para ello separa un 20%
% de los datos para validación, y encuentre el mejor valor para el parámetro
% de regularización. Re-entrena con todos los datos para el mejor valor de 
% landa, y utiliza los datos de test para calcular la matriz de confusión y 
% los valores de precisión y recall para cada dígito, y visualiza las 
% confusiones con la función verConfusiones. ¿Qué dígitos son los más problemáticos?
nc = 10; % numero de clases
probar_lambdas = logspace(-20, 1, 20); % lambdas a probar (500 entre 10^-20 y 1)
ingenuo = 1; % con bayes ingenuo
k = 5; % 5 folds para que separe cada vez el 20% para validación
[modelo, best_lambda] = kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas);
% kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas)
best_lambda

%% 3. Clasificar 
% se clasifican los datos de test a partir de los modelos aprendidos con
% los datos de entrenamiento
yhattest = clasificacionBayesiana(modelo, Xtest);

%% 3. Evaluar
% Se evalúan, comparando las predicciones con las clases reales de cada
% dato
verConfusiones(Xtest, ytest, yhattest); % Ejemplos de confusiones
matriz_conf = confusionmat(ytest, yhattest); % Matriz de confusiones
confusionchart(ytest, yhattest); % Misma matriz, pero visual
% obtenemos la precision y recall para cada digito a partir de la matriz:
[precision, recall] = precision_recall(matriz_conf)



%% 4. Entrenar Bayes no ingenuo
% Repite el apartado anterior en el caso general con matrices de covarianzas completas, 
% y compara con los resultados obtenidos con Bayes ingenuo. ¿Cual de los dos modelos funciona mejor?
% Compara también con los resultados de la práctica anterior.
nc = 10; % numero de clases
probar_lambdas = logspace(-6, 0, 20); % lambdas a probar (20 entre 10^-20 y 10)
ingenuo = 0; % con bayes no ingenuo
k = 5; % 5 folds para que separe cada vez el 20% para validación
[modelo, best_lambda_no_ingenuo] = kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas);
% kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas)
best_lambda_no_ingenuo

%% 4. Clasificar 
% se clasifican los datos de test a partir de los modelos aprendidos con
% los datos de entrenamiento
yhattest = clasificacionBayesiana(modelo, Xtest);

%% 4. Evaluar
% Se evalúan, comparando las predicciones con las clases reales de cada
% dato
verConfusiones(Xtest, ytest, yhattest); % Ejemplos de confusiones
matriz_conf = confusionmat(ytest, yhattest); % Matriz de confusiones
confusionchart(ytest, yhattest); % Misma matriz, pero visual
% obtenemos la precision y recall para cada digito a partir de la matriz:
[precision, recall] = precision_recall(matriz_conf)



