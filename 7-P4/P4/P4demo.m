%% Practica 4 
% Based on exercise 3 of Machine Learning Online Class by Andrew Ng 
%

clear ; close all;
addpath(genpath('../minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);


%% kfold multiclase:
disp('--------------------------------------------------------');
disp('kfold regularizado multiclase:')
k = 5;
probar_lambdas = logspace(-8, 0, 10); % 10 lambdas entre 10^-8 y 10^0
% Xtr_exp = mapFeature(X_train(:,1), X_train(:,2));
[theta, lambda, yhat, tasa] = kfold_logistica_multi(X,y,k,probar_lambdas);
lambda, tasa


%% confusiones train:
verConfusiones(X, y, yhat);
confusionchart(y, yhat);
confusiones = confusionmat(y, yhat);

[precision, recall, F1_score] = precision_recall(confusiones)
%% confusiones test
yhat_test = clasificar_multi(Xtest, theta);

verConfusiones(Xtest, ytest, yhat_test);
confusionchart(ytest, yhat_test);

confusiones_test = confusionmat(ytest, yhat_test);

[precision, recall, F1_score] = precision_recall(confusiones_test)


%% intento de ajustar clase 8
clase_excesiva = 8; % el clasificador de esta clase es demasiado optimista
penalizacion = 0.1; % se le restara esta probabilidad antes de la clasificacion multi

yhat_test = clasificar_multi_ajustado(Xtest, theta, clase_excesiva, penalizacion);

% verConfusiones(Xtest, ytest, yhat_test);
confusionchart(ytest, yhat_test);

confusiones_test = confusionmat(ytest, yhat_test);

[precision, recall, F1_score] = precision_recall(confusiones_test)


%% bucle para maximizar F1
clase_excesiva = 8; % el clasificador de esta clase es demasiado optimista

best_F1 = 0;
for penalizacion = 0:0.01:1
    yhat_train = clasificar_multi_ajustado(X, theta, clase_excesiva, penalizacion);
    confusiones_test = confusionmat(y, yhat_train);
    [precision, recall, F1_score] = precision_recall(confusiones_test);
    if F1_score > best_F1
        best_F1 = F1_score;
        best_penalizacion = penalizacion;
    end
end
best_F1, best_penalizacion


% nueva penalizacion en los datos de test:
yhat_test = clasificar_multi_ajustado(Xtest, theta, clase_excesiva, best_penalizacion);
confusiones_test = confusionmat(ytest, yhat_test);
[precision, recall, F1_score] = precision_recall(confusiones_test)

confusionchart(ytest, yhat_test);

