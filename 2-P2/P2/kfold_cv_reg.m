function [hypothesis, best_lambda, h_mu, h_sd] = kfold_cv_reg(k, X, y, grados, lambda_vals)
%	Prueba los con parametros de regularizacion en lambda_vals y devuelve el mejor entrenamiento


    best_lambda = 0;
    best_errV = inf;
    valores_err_T = [];
    valores_err_V = [];
    
    X_exp = expandir(X, grados); % expansion
    for i = 1:size(lambda_vals,2)
        % para cada lambda:
        lambda = lambda_vals(i);
        
        err_T = 0;
        err_V = 0;
        for fold = 1:k 
            % separar N/k ejemplos para validación
            
            [Xval, yval, Xtr, ytr] = particion(fold, k, X_exp, y); % function [ Xcv, ycv, Xtr, ytr ] = particion( i, nfolds, X, y )
            
            [X_exp_n, mu, sd] = normalizar(Xtr); % normalizacion
            
            % aqui, D=size(Xtr,2)-1 es la dimension vertical - 1, ya que
            % el th0 no se penaliza
            H = X_exp_n'*X_exp_n + lambda*diag([0 ones(1,size(X_exp_n,2)-1)]); 
            hypo_w = H \ (X_exp_n'*ytr); % esta normalizado, hay que desnormalizarlo
            hypo_w_des = desnormalizar(hypo_w, mu, sd);
            
            err_T = err_T + RMSE( hypo_w_des, Xtr, ytr ); % Error en train
            err_V = err_V + RMSE( hypo_w_des, Xval, yval); % Error en val
        end
        % Promedios:
        err_T = err_T/k;  err_V= err_V/k; % guardamos estos para la grafica:
        valores_err_T = [valores_err_T; err_T];
        valores_err_V = [valores_err_V; err_V];
        %{calcular el error medio de las k veces}
        if err_V < best_errV %&& has_converged(err_T) % TODO: has_converged??
            mejor_X_exp_n = X_exp_n;
            mejor_lambda_tmp = lambda;
            best_errV = err_V; % Guardar mejor error de validacion (faltaba en la transparencia)
            h_mu_tmp = mu;
            h_sd_tmp = sd;
            
            % {guardar el mejor valor de los hyper-parámetros}
        end
    end
    % grafica de evolucion de errores
    plot_errores_reg(valores_err_T, valores_err_V, lambda_vals, 'log(lambda)', 'RMSE - lambda');
    % outputs, los mejores del conjunto:
    best_lambda = mejor_lambda_tmp;
    h_mu = h_mu_tmp;
    h_sd = h_sd_tmp;
    
    [X_exp_n, mu, sd] = normalizar(X_exp);
    H = X_exp_n'*X_exp_n + best_lambda*diag([0 ones(1,size(X_exp_n,2)-1)]); 
    hypo_w = H \ (X_exp_n'*y); % esta normalizado, hay que desnormalizarlo
    hypothesis = desnormalizar(hypo_w, mu, sd);
    
end