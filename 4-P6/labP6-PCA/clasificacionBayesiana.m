function yhat = clasificacionBayesiana( modelo, X, clases)
% Con los modelos entrenados, predice la clase para cada muestra X

% yhat es la clase mas probable para cada muestra de X
% Se calcula de la siguiente forma, utilizando la distancia de mahalanobis:
% yhat = argmax( sigma{clase}^(-1/2) exp(dist_mahalanobis(clase,muestra)) * p(clase) )
%      = argmax( -1/2 * ln(sigma{clase}) - 1/2(dist_mahalanobis(clase,muestra) + ln(p(clase)))  )
% se utiliza la segunda forma ya que es equivalente y es computacionalmente
% menos costosa, al no necesitar invertir la matriz sigma
% en nuestro caso, como las clases son equiprobables, también podemos
% ahorrarnos el último término (ln(p(clase)))

% num de clases, tamaño del modelo:
nclases = size(modelo,2);
logp_clases = [];
for clase = clases % para cada clase
    % Primero, los logaritmos:
    logp = gaussLog(modelo{clase}.mu, modelo{clase}.sigma, X); % logp = gaussLog(mu, Sigma, X)
    logp_clases = [logp_clases logp];
end

% [valor, indice] (nos interesa el indice, es la clase)
[~,i] = (max(logp_clases,[],2));
yhat = clases(i)';