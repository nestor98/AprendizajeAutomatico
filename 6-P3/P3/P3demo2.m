%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% Load and Plot Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('mchip_data.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
N = length(y);
p = randperm(N); %reordena aleatoriamente los datos
X = X(p,:);
y = y(p);

plotData(X, y);
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')


%% Dividir en train, test:
porcentaje = 0.2; % porcentaje de muestras para test
i = 1/porcentaje; % inversa, para la funcion particion
% particion, usada en el kfold de la practica 3:
[X_test, y_test, X_train, y_train] = particion(randi(i), i, X, y);
N_train = size(y_train,1);

%% kfold sin normalizar:
disp('--------------------------------------------------------');
disp('kfold regularizado sin normalizar Xexp:')
k = 5;
probar_lambdas = logspace(-15, 0, 50); % 50 lambdas entre 10^-20 y 10^0
Xtr_exp = mapFeature(X_train(:,1), X_train(:,2));
[theta, lambda, tasa] = kfold_logistica(Xtr_exp,y_train,k,probar_lambdas);
lambda, tasa

modelo.theta = theta;
modelo.lambda = lambda;
modelo.tasa_tr = tasa;
modelos = [modelo];


%% kfold normalizado
disp('--------------------------------------------------------');
disp('kfold regularizado normalizando Xexp:')
k = 5; % numero de folds
probar_lambdas = logspace(-15, 0, 50); % 50 lambdas entre 10^-20 y 10^0
Xtr_exp = mapFeature(X_train(:,1), X_train(:,2));
X_train_n = normalizar(Xtr_exp);

[theta, lambda, tasa] = kfold_logistica(X_train_n,y_train,k,probar_lambdas);
lambda, tasa
modelo.theta = theta;
modelo.lambda = lambda;
modelo.tasa_tr = tasa;
modelos(end+1) = modelo;

%% Modelo con lambda = 0
disp('--------------------------------------------------------');
disp('tasa de errores usando lambda=0:')
 % opciones de minFunc:
options = [];
options.method = 'Newton';
options.display = 'none'; 

theta = zeros(size(Xtr_exp,2),1);
th = minFunc(@costeLogReg,theta,options,Xtr_exp,y_train,0);
yhat = clasificar(Xtr_exp, th);
tasa_lambda0 = tasa_errores(yhat, y_train)

modelo.theta = th;
modelo.lambda = 0;
modelo.tasa_tr = tasa_lambda0;
modelos(end+1) = modelo;

%% tasas de errores con test:
X_test_exp = mapFeature(X_test(:,1), X_test(:,2));

% kfold, sin normalizar
modelos(1).yhat = clasificar(X_test_exp, modelos(1).theta);
modelos(1).tasa_test = tasa_errores(modelos(1).yhat, y_test);

% kfold, normalizado:
X_test_n = normalizar(X_test_exp);
modelos(2).yhat = clasificar(X_test_n, modelos(2).theta);
modelos(2).tasa_test = tasa_errores(modelos(2).yhat, y_test);

% lambda = 0:
modelos(3).yhat = clasificar(X_test_exp, modelos(3).theta);
modelos(3).tasa_test = tasa_errores(modelos(3).yhat, y_test);

disp('--------------------------------------------------------');
disp('tasas de errores con datos de test de cada modelo:')
modelos.tasa_test

% abandonamos el segundo modelo, el normalizado, ya que es 
% equivalente al primero

%% dibujamos las curvas de separacion:

X_exp = mapFeature(X(:,1), X(:,2));

% dibujamos los dos modelos interesantes (1 y 3)
for i = [1 3]
    plotDecisionBoundary(modelos(i).theta, X_exp, y);
    title(sprintf('lambda = %g', modelos(i).lambda))
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend('y = 1', 'y = 0', 'Decision boundary')
end

%% matriz de confusion
confusiones = confusionmat(modelos(1).yhat, y_test);
confusionchart(modelos(1).yhat, y_test);
[precision, recall, F1_score] = precision_recall(confusiones)

%% con lambda0
confusiones=confusionmat(modelos(3).yhat, y_test);
confusionchart(modelos(3).yhat, y_test);
[precision, recall, F1_score] = precision_recall(confusiones)


%% 95% de los aceptados sean buenos
% es decir, que el 95% de los clasificados como buenos sean buenos
% -> los buenos son la clase 0: precision del 0.95 en la clase 0
disp('--------------------------------------------------------');
disp('Umbral tal que el 0.95 de los aceptados sean buenos')
i_clase = 1; % (la clase 0 tiene indice 1 en la matriz)
[yhat, umbral, confusiones] = umbral_precision(Xtr_exp, y_train, modelos(1).theta, i_clase, 0.95);
umbral, confusiones
confusionchart(yhat, y_train);
[precision, recall, F1_score] = precision_recall(confusiones)

%% aplicandolo a los datos de test:
disp('--------------------------------------------------------');
disp('confusiones en los datos de test con el umbral calculado');
yhat_test = clasificar(X_test_exp, modelos(1).theta, umbral);
confusiones_test = confusionmat(yhat_test, y_test);
confusionchart(yhat_test, y_test);

[precision, recall, F1_score] = precision_recall(confusiones_test)



