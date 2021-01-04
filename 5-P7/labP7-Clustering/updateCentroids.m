function munew = updateCentroids(D, c)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

munew = [];

for i = unique(c)' % para cada clase en c
    % media de los valores de D cuyos indices en c son i
    mu = mean(D(c==i,:));
    munew = [munew; mu]; 
end