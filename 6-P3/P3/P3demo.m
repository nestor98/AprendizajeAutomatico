%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 

clear ; close all;

%% otro load
% load('exam_data.txt');
% 
% load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
% rand('state',0);
% p = randperm(length(y));
% X = X(p,:);
% y = y(p);

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('exam_data.txt');
y = data(:, 3);
N = length(y);
X = data(:, [1, 2]); 

plotData(X, y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')

%% Dividir en train, test:
porcentaje = 0.2; % porcentaje de muestras para test
i = 1/porcentaje; % inversa, para la funcion particion
% particion, usada en el kfold de la practica 3:
[X_test, y_test, X_train, y_train] = particion(randi(i), i, X, y);
N_train = size(y_train,1);

%% Calculo de theta por descenso de gradiente (por probar):
% X = [ones(N,1) X];
theta = [-70 0.7 0.3]';  
X_train_ones = [ones(N_train,1) X_train];
[X_train_n, mu, sd] = normalizar(X_train_ones);
J = [];

alfa = 0.8;

% mientras el penultimo J sea mayor que el ultimo
while (size(J,1) < 2 || J(end-1) > J(end)) 
    [J_act,grad] = costeLogistico(theta,X_train_n,y_train);
    theta = theta - alfa .* grad;
    J = [J; J_act];
end
plotJ(J, "Coste J por iteracion", "Iteracion");
disp('J final:');
J(end-1)
J(end) % siempre acaba siendo NaN al dividirse 0/0 o multiplicarse log(0)*0

%% 2 - calcular theta con minFunc
disp('--------- 2: Regresión logística básica ---------');
 
theta = [-70 0.7 0.3]'; % no importan los valores iniciales

X_train_ones = [ones(N_train,1) X_train];
[X_train_n, mu, sd] = normalizar(X_train_ones);
options = [];
options.method = 'Newton';
options.display = 'final';

theta = minFunc(@costeLogistico,theta,options,X_train_n,y_train);
J = costeLogistico(theta,X_train_n,y_train)

%% Tasas de errores
% Train:
% Clasificar y calcular tasa de errores:
yhat = clasificar(X_train_n, theta);
tasa_train = tasa_errores(yhat, y_train)

% Test:
% Añadir columna de unos y normalizar:
X_test_ones = [ones(size(X_test,1),1) X_test];
[X_test_n] = normalizar(X_test_ones);
% Clasificar y calcular tasa de errores:
yhat = clasificar(X_test_n, theta);
tasa_test = tasa_errores(yhat, y_test)

%% Probabilidad de aprobar de un alumno con 45 en el primer examen:
n_puntos = 100; % numero de distintas probabilidades a calcular
% posibles notas del alumno (primera 45, segunda de 1 a 100):
X_posibles = [repmat(45, n_puntos, 1) (1:n_puntos)']

X_pos_ones = [ones(size(X_posibles,1),1) X_posibles];
% Esta vez normalizamos con mu y sd de train, ya que en este caso 
% las de estos datos serian muy distintas
X_pos_n = normalizar(X_pos_ones, mu, sd);
prob = 1./(1+exp(-(X_pos_n*theta)));

% plot de la probabilidad en funcion de la nota del segundo examen:
plot_prob(X_posibles(:,2), prob, "Probabilidad de aprobar según el segundo examen", "Probabilidad", "Nota en el segundo examen");

% en forma de tabla:
tabla = [X_posibles(:,2), prob]