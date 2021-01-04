%% Practica 6.2: PCA 

clear all
close all

% Leer la imagen
I = imread('mi cara_3.jpg');

% Convertirla a blanco y negro
BW = rgb2gray(I);

% Convertir los datos a double
X=im2double(BW);

% graficar la imagen
figure(1);
colormap(gray);
imshow(X);
axis off;
pause

% Estandarización de los datos: -> no es necesaria para SVD
% [Xn, mu] = estandarizar(X);
% % graficar la imagen centrada
% figure(2);
% colormap(gray);
% imshow(Xn);
% axis off;
% pause

%%%%%%%%%%%%%%%% Aplicar PCA (SVD): %%%%%%%%%%%%%%%%

%% matrices U, S (sigma) y V:
[U,S,V] = svd(X);
% svd ya las devuelve ordenadas por valores singulares:
anterior = S(1,1)
ordenadas = 1;
for i = 2:size(S,2)
    actual = S(i,i);
    ordenadas = (anterior >= actual);
    if ~ordenadas
        disp('no estan ordenados');
        break;
    else
        anterior = actual;
    end
end
ordenadas


%% Graficar las primeras 5 componentes
for k = 1:5
    %[Xhat, Xhat2] = proyectarSVD(U, S, V, k);
    % reconstruimos la imagen de la siguiente forma, a partir de 
    % los vectores singulares (con V transpuesta):
    Xhat = (U(:,1:k) * S(1:k,1:k) * V(:, 1:k)');
    figure(2+k);
    % Deshacemos la estandarización sumandole la media de nuevo:
    imshow(Xhat);
    colormap(gray);
    axis off;
    pause
end

%% Graficar la reconstrucción con las primeras 1, 2, 5, 10, 20, y total
% de componentes
for k = [1 2 5 10 20 rank(X)]
    figure(7+k);
    Xhat = (U(:,1:k) * S(1:k,1:k) * V(:, 1:k)');
    imshow(Xhat);
    colormap(gray);
    axis off;
    pause
end

% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
%% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad

disp('Para mantener el 90% de la variabilidad, el menor valor de k (y su variabilidad real) es:')
[k, varK] = obtenerK(diag(S), 0.9)


%% Graficar la reconsrtucción con las primeras k componentes
figure(14);
Xhat = (U(:,1:k) * S(1:k,1:k) * V(:, 1:k)');
imshow(Xhat);
colormap(gray);
axis off;
pause



%% Calcular y mostrar el ahorro en espacio


% dimensiones de los datos originales:
m = size(X,1);
n = size(X,2);
% Grafico:
figure(15);
title('Ratio de compresión');
grid on;
hold on;
plot((m * n) ./ ((1:n) * (m+n+1)));
%plot(((1:n) * (m+n+1)) ./ (m * n));
xlabel('k');
ylabel('Ratio (original/comprimido)');


%% En el caso del k=43:
disp('ahorro de espacio con k=43:')
ratio = (m*n) / (k * (m+n+1))

