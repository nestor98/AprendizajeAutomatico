function [hypothesis, best_lambda] = kfoldcv_gaussianas(X, y, nc, ingenuo, k, probar_lambdas)%;(k, X, y,lambda_vals)
%	Prueba los parametros de regularizacion en probar_lambdas y devuelve el
%	mejor entrenamiento posible con la mejor lambda segun los k folds


%   best_lambda = 0;
    best_errV = inf;
    valores_err_T = [];
    valores_err_V = [];
    
    
    for i = 1:size(probar_lambdas,2)
        % para cada lambda:
        lambda = probar_lambdas(i);
        
        err_T = 0;
        err_V = 0;
        for fold = 1:k 
            % separar N/k ejemplos para validación
            
            [Xval, yval, Xtr, ytr] = particion(fold, k, X, y); % function [ Xcv, ycv, Xtr, ytr ] = particion( i, nfolds, X, y )
            
            % entrenamos:
        
            modelo = entrenarGaussianas(Xtr, ytr, nc, ingenuo, lambda);
            % evaluamos el fold. Al tener el mismo num de muestras de cada clase (clases no sesgadas), 
            % podemos utilizar la tasa de errores directamente. Si no fuera el caso, deberiamos calcular
            % precision y recall y podriamos emplear su media geometrica, por ej:
            yhat_tr = clasificacionBayesiana(modelo, Xtr);
            err_T = err_T + tasa_errores(yhat_tr, ytr); % Error en train
            
            yhat_val = clasificacionBayesiana(modelo, Xval);      
            err_V = err_V + tasa_errores(yhat_val, yval); % Error en val
        end
        % Promedios:
        err_T = err_T/k;  err_V= err_V/k; % guardamos estos para la grafica:
        valores_err_T = [valores_err_T; err_T];
        valores_err_V = [valores_err_V; err_V];
        %{calcular el error medio de las k veces}
        if err_V < best_errV %&& has_converged(err_T) 
            mejor_lambda_tmp = lambda;
            best_errV = err_V; % Guardar mejor error de validacion  
            % {guardar el mejor valor de los hyper-parámetros}
        end
    end
    % grafica de evolucion de errores
    plot_errores_reg(valores_err_T, valores_err_V, probar_lambdas, 'log(lambda)', 'tasa errores', 'tasa errores - lambda');
    % outputs, los mejores del conjunto:
    best_lambda = mejor_lambda_tmp;

    
    hypothesis = entrenarGaussianas(X, y, nc, ingenuo, best_lambda);
end