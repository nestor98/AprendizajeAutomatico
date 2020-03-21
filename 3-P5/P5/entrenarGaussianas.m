function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza ser�n diagonales
% Se regularizar�n las covarianzas mediante: Sigma = Sigma + landa*eye(D)

