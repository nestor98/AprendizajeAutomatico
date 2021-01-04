function [precision, recall, F1_score] = precision_recall(conf_mat)
% devuelve los valores de precision y recall a partir de la
% matriz de confusiones <conf_mat>
% true_pos = sum(y==yhat==clase);
% false_pos = sum(yhat==clase && yhat~=y); % coinciden con la clase pero no con la verdad
% false_neg = sum(y==clase && yhat~=y); % es un positivo pero se clasifica como negativo



accuracy = 1;%conf_mat()sum(y==yhat);
% basado en https://es.mathworks.com/matlabcentral/answers/486122-precision-recall-perfcurve
precision = diag(conf_mat)./sum(conf_mat,1)';
recall = diag(conf_mat)./sum(conf_mat,2);

% basado en https://es.mathworks.com/matlabcentral/answers/262033-how-to-calculate-recall-and-precision
F1_score = 2*mean(recall)*mean(precision)/(mean(precision)+mean(recall));
end

