function [] = dibujarVect(U,L)
% Dibuja el (i_vector)-ésimo vector de U * 3 veces la raíz cuadrada de su valor
% propio (en L(i_vector,i_vector))
figure(2); % para que se superpongan a la nube de datos
axis equal;
grid on;
hold on;
%plot3(U(1,:));
v = U(:,1:3) * 3*sqrt(L(1:3,1:3));

v2 =  U(:,2) * 3*sqrt(L(2,2));

v3 = U * 3*sqrt(L);
% 
% v2 =  U(:,i_vector+1) * 3*sqrt(L(i_vector+1,i_vector+1));

quiver3(0,0,0,v(1, 1),v(2, 1),v(3,1 ), 'r-', 'LineWidth', 2);
quiver3(0,0,0,v(1,2),v(2, 2),v(3,2), 'r-', 'LineWidth', 2);
quiver3(0,0,0,v(1, 3),v(2, 3),v(3, 3), 'r-', 'LineWidth', 2);
% quiver3(0,0,0,v2(1),v2(2),v2(3), 'r-');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');

legend('vectores propios')
pause
end
