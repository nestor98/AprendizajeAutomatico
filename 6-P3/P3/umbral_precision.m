function [yhat, umbral, confusiones] = umbral_precision(X, y, theta, clase, precision_buscada)
% devuelve yhat y umbral correspondientes al menor umbral tal que la clase
% <clase> tenga al menos <precision_buscada> de precision
mejor_precision = -1;
umbrales = 0:0.01:1; % de 0 a 1 en pasos de 0.01

if clase == 2 % si es la segunda clase hay que ir bajando el umbral en lugar de subirlo
    umbrales = flip(umbrales);
end

i=1;

while (mejor_precision < precision_buscada)
    umbral = umbrales(i);
    yhat = clasificar(X, theta, umbral);
    confusiones=confusionmat(yhat, y);
    [precision, recall, F1_score] = precision_recall(confusiones);
    mejor_precision = precision(clase);
    
    i = i+1;
    if i > 100
       break 
    end
end


end

