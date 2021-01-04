function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
% Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)

% cada clase es una de las cifras que queremos distinguir 0..9

% nc numero de clases
for clase = 1:nc
    modelo{clase}.N = nMuestrasClase(ytr, clase);
    
    Xclase = Xtr(ytr==clase, :); % Xtr de la clase
    modelo{clase}.mu = mean(Xclase, 1); % medias de las columnas (pixeles)
    if NaiveBayes == 0
        % con bayes no ingenuo
        modelo{clase}.sigma = cov(Xclase); % Matriz de covarianzas
        
%         modelo{clase}.sigma(modelo{clase}.sigma < 0) = 0; % arreglar error por el que salen ciertas covarianzas negativas
    else
        % bayes ingenuo
        varianzas = var(Xclase);
        modelo{clase}.sigma = diag(varianzas); % como matriz diagonal
    end
    % regularizar (Sigma = Sigma + landa*eye(D)):
    Dclase = size(Xclase,2);
    modelo{clase}.sigma = modelo{clase}.sigma + landa*eye(Dclase);
end

