function [th, best_lambda, best_tasa] = kfold_logistica(X, y, k, probar_lambdas)
%	Prueba los parametros de regularizacion en probar_lambdas y devuelve el
%	mejor entrenamiento posible con la mejor lambda segun los k folds
%   (funcion basada en kfoldcv_gaussianas, de la practica 5)

    best_errV = inf;
    valores_err_T = [];
    valores_err_V = [];
    % opciones de minFunc:
    options = [];
    options.method = 'Newton';
    options.display = 'none'; % sin display, que hay muchas iteraciones

    for lambda = probar_lambdas
        % para cada lambda:
        
        err_T = 0;
        err_V = 0;
        for fold = 1:k 
            % separar N/k ejemplos para validación
            
            [Xval, yval, Xtr, ytr] = particion(fold, k, X, y); % function [ Xcv, ycv, Xtr, ytr ] = particion( i, nfolds, X, y )
            
            % Ampliamos Xtr con la expansion de funciones base:
%             Xtr = mapFeature(Xtr(:,1), Xtr(:,2));
            
            % Inicializar theta:
            theta = zeros(size(Xtr,2),1); % inicial: columna de (numero de columnas de Xtr) ceros
            % entrenamiento:
            theta = minFunc(@costeLogReg,theta,options,Xtr,ytr,lambda);
%             J = costeLogistico(theta,X_train_n,y_train)
            % clasificacion y tasa de errores de train:
            yhat = clasificar(Xtr, theta);
            err_T = err_T + tasa_errores(yhat, ytr);
            % clasificacion y tasa de errores de val
            yhat = clasificar(Xval, theta);
            err_V = err_V + tasa_errores(yhat, yval);
        end
        % Promedios:
        err_T = err_T/k;  err_V= err_V/k; % guardamos estos para la grafica:
        valores_err_T(end+1) = err_T;
        valores_err_V(end+1) = err_V;
        %{calcular el error medio de las k veces}
        if err_V < best_errV % si el de val es el mejor, lo guardamos
            mejor_lambda_tmp = lambda;
            best_errV = err_V; % Guardar mejor error de validacion  
            % {guardar el mejor valor de los hyper-parámetros}
        end
    end
    % grafica de evolucion de errores
    plot_errores_reg(valores_err_T, valores_err_V, probar_lambdas, 'log(lambda)', 'tasa errores', 'tasa errores - lambda');
    % outputs, los mejores del conjunto:
    best_lambda = mejor_lambda_tmp;
    % devolvemos el entrenamiento final con best_lambda:
    th = minFunc(@costeLogReg,theta,options,X,y,best_lambda);
    yhat = clasificar(X, th);
    best_tasa = tasa_errores(yhat, y);
end