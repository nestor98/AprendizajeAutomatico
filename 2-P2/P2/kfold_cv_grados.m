function [hypothesis, mejor_grado, h_mu, h_sd] = kfold_cv_grados(k, X, y, i, grados)
%	Prueba los entre 1 y grados(i) en grados y devuelve el mejor entrenamiento
%   Por ejemplo: si X = [x1 x2 x3], i=1, y grados = [3 1 1], devuelve el mejor entre: 
%   Xexp = [1 x1 x2 x3]; [1 x1 x1^2 x2 x3] y [1 x1 x1^2 x1^3 x2 x3]

    mejor_grado_tmp = 1;
    best_errV = inf;
    grados_iter = grados; % grados a utilizar
    valores_err_T = [];
    valores_err_V = [];
    for grado = 1:grados(i)
        % para cada grado
        grados_iter(i) = grado; % se sustituye el buscado por el de la iteracion

        X_exp = expandir(X, grados_iter); % expansion
%         X_exp_n=[];mu=0;sd=0;
        
        [X_exp_n, mu, sd] = normalizar(X_exp); % normalizacion
        
        err_T = 0;
        err_V = 0;
        for fold = 1:k 
            % separar N/k ejemplos para validación
            
            [Xval, yval, Xtr, ytr] = particion(fold, k, X_exp_n, y); % function [ Xcv, ycv, Xtr, ytr ] = particion( i, nfolds, X, y )
            
            hypo_w = Xtr\ytr; % pesos de hipotesis
            % {aprender con el resto}
            err_T = err_T + RMSE( hypo_w, Xtr, ytr ); % Error en train
            err_V = err_V + RMSE( hypo_w, Xval, yval); % Error en val
        end
        % Promedios:
        err_T = err_T/k;  err_V= err_V/k; % guardamos estos para la grafica:
        valores_err_T = [valores_err_T; err_T];
        valores_err_V = [valores_err_V; err_V];
        %{calcular el error medio de las k veces}
        if err_V < best_errV %&& has_converged(err_T) % TODO: has_converged??
            mejor_X_exp_n = X_exp_n;
            mejor_grado_tmp = grado;
            best_errV = err_V; % Guardar mejor error de validacion (faltaba en la transparencia)
            h_mu_tmp = mu;
            h_sd_tmp = sd;
            
            % {guardar el mejor valor de los hyper-parámetros}
        end
    end
    % grafica de evolucion de errores
    plot_errores(valores_err_T, valores_err_V, 1:grados(i), 'grados', 'RMSE - grado');
    % outputs, los mejores del conjunto:
    mejor_grado = mejor_grado_tmp;
    h_mu = h_mu_tmp;
    h_sd = h_sd_tmp;
    hypothesis_n = mejor_X_exp_n\y;%Learner(X);   %{aprender de nuevo con todos}
    hypothesis = desnormalizar(hypothesis_n, h_mu,h_sd);
end


% Funcion original de las transparencias:
% Function kfold_cross_validation(Learner, k, examples) returns hypothesis 
% best_size ? 0;  
% best_errV ? inf; 
% for size = 1 to n 
%     do{para los distintos valores de los hyper-parámetros}
%     err_T ? 0;  
%     err_V? 0  
%     for fold = 1 to k 
%         do{separar N/k ejemplos para validación}
%         [training_set, validation_set] ? Partition(examples, fold, k)  
%         h ? Learner(size, training_set)
%         {aprender con el resto}
%         err_T ? err_T + Error(h, training_set)
%         err_V ? err_V + Error(h, validation_set)
%     end for
%     err_T ? err_T/k;  err_V? err_V/k
%     {calcular el error medio de las k veces}
%     if has_converged(err_T) and err_V < best_errV 
%         then 
%         best_size ? size
%         {guardar el mejor valor de los hyper-parámetros}
%     end
% end 
%     return Learner(best_size, examples)   %{aprender de nuevo con todos}