clc, clear variables, close all

thickness = 80;     %These three are the only parameters you need
philength = 16384;   % to set.
numofbits = 10;     %Philength is the resolution in the circumferencial
                    % direction, and thickness is the thickness in radial
                    % direction.
                    %Note that philength/(2^numofbits) must be an integer
                    % otherwise, the code will fail due to indexing issues
                    % So if you for example have 4 bits, 2^4 = 16, you could
                    % then choose a philength of 3200, 3200/16 = 200, which
                    % is fine.
                                                      
%---------------------------------------------
%The following code is not necessary to change
% but feel free to do so if you wish to
%---------------------------------------------
 X = 0:(2^numofbits - 1);
 y = bin2gray(X,'pam',(2^numofbits));
 
 G = de2bi(y);
 G = fliplr(G);
 G = G';

 r = 1:thickness;
storlek = size(G);
 Gheight = storlek(1);
 Glength = storlek(2);

numofnum = 2^numofbits;
 siz = thickness+Gheight*thickness+1;

phi = linspace(0,2*pi,philength);
A = [];
N = [];

for i = 1:Gheight
    for j = 1:numofnum
        for k = 1:(philength/numofnum)
            
            N(i,((j-1)*philength/numofnum+k)) = G(i,j);
        
        end
    end
end

for i = 1:Gheight
    for j = 1:thickness
       for k = 1:philength   
    
            x = (r(j)+i*thickness)*cos(phi(k))+siz;
            y = (r(j)+i*thickness)*sin(phi(k))+siz;
   
            x = round(x);
            y = round(y);
    
            if N(i,k) == 0
                A(x,y) = 255; 
            end
            if N(i,k) == 1
                A(x,y) = 0;
            end
       
        end
    end
end

A = abs(A - 255); %This inverts the colors, comment if you want the other 
                  % color combination

I = mat2gray(A);

imshow(I)
 

 