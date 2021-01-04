figure(1)
% im = imread('smallparrot.jpg');
im = imread('atardecer_red.jpg');

imshow(im)

%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));

%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans 
K = 1000;

%% Inicializar centroides
% para evitar tomar dos centroides iguales:
Dunique = unique(D, 'rows'); % filas unicas de D
i = randsample(1:size(Dunique,1), K); % indices de filas aleatorias de Dunique
mu0 = Dunique(i,:); % mu0 con K filas aleatorias y unicas de D
%ismember(mu0, D, 'rows') % para comprobar que mu0 pertenece a D

%% bucle kmeans
[mu, c, J] = kmeans(D, mu0, 1);

disp('valor de J:');
J

%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(3)
imshow(uint8(qIM));


%% Optimizar k:
% Para almacenar las Js, mus y clases de cada valor de K:
Js = []; 
mus = [];
cs = [];
for K = 2:10
    i = randsample(1:size(Dunique,1), K); % indices de filas aleatorias de Dunique
    mu0 = Dunique(i,:); % mu0 con K filas aleatorias y unicas de D
    [mu, c, J] = kmeans(D, mu0, 0);
    Js = [Js; J];
    mus = [mus; mu];
    cs = [cs; c];
end

plotJ(Js,'J en función de K', 'K');

%% reconstruir imagen con el nuevo valor de K:
K = 30

qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure(3)
imshow(uint8(qIM));




%% ahorro de espacio
% dimensiones de los datos originales:
m = size(D,1);
n = size(D,2);
% Grafico:
figure(15);
title('Ratio de compresión');
grid on;
hold on;
% necesitamos guardar c(m,1) y mu(k,n) -> m+k*n
k = 1:m;
tam_k = m + k*n;
plot((m * n) ./ tam_k);
%plot(((1:n) * (m+n+1)) ./ (m * n));
xlabel('k');
ylabel('Ratio (original/comprimido)');


%% En el caso del k=15:
k=1000
disp('ahorro de espacio con k=1000:')
ratio = (m*n) / (m + k * n)


