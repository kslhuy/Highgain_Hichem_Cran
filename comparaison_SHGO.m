clear all 
close all
clc
rng(97)
%System 
L_f=2.71;
%v=4;
n= 6;
a=[zeros(2,1) eye(2) ;zeros(1,n/2)];
A=blkdiag(a,a);
c=[1 zeros(1,n/2-1)];
C= blkdiag(c,c);
b=[zeros(n/2-1,1); 1];
B=blkdiag(b,b);
%Simulation
load('Data.mat')
sim_window=1:350;
tspan = time(sim_window);


%LMI

Sigma1=[5,5, 5];
Sigma2=[-5,-5, -5];
lambda = 10^7;
P= sdpvar(n/2,n/2) ; 
Y=sdpvar(1,n/2); 
Z=sdpvar(1,n/2); 
M1= [ a'*P + P*a - c'*Y - Y'*c-Sigma1'*Z-Z'*Sigma1 + lambda * eye(n/2), zeros(n/2,n/2);
    zeros(n/2,n/2), -P ];
M2= [ a'*P + P*a - c'*Y - Y'*c-Sigma2'*Z-Z'*Sigma2 + lambda * eye(n/2), zeros(n/2,n/2);
    zeros(n/2,n/2), -P ];
const = [M1 <= 0;M2<=0;Z>=10000 ;];
diagnostic=optimize(const,[]);
if diagnostic.problem ~= 0
    clc
    error(diagnostic.info)
end
clc
K=value(P)\value(Y)';
%K=  [1.9633;
 %   4.0861;         
%    1.4985];
M=value(P)\value(Z)';
K=[K zeros(n/2,1);zeros(n/2,1) K];
K_=K*0.1;
M=blkdiag(M,M);
M_=M*10;
%%
x(:,1:2)=position(:,sim_window)';
v=sqrt(velocity(1,sim_window).^2+velocity(2,sim_window).^2);
vxy=velocity(:,sim_window);
x(:,3)=yaw(sim_window)-yaw(1);
Yaw_rate=yaw_rate(sim_window);

%Observateur Grand gain 
%calcul de thetha 
syms z1 z2 z3 z4 z5 z6 
f1= -1/(z2^2+z5^2)*(-z5*z3+z2*z6)*z6;
f2= 1/(z2^2+z5^2)*(-z5*z3+z2*z6)*z3;
J=jacobian([f1,f2],[z1, z2, z3, z4, z5, z6 ]);
syms f(z1, z2, z3, z4, z5, z6)
f(z1, z2, z3, z4, z5, z6)=norm(J);
z1max= 250;
z2max= 12;
z3max= 2;
z4max= z1max;
z5max= z2max;
z6max= z3max;
L=double(f(z1max, z2max,z3max,z4max, z5max, z6max));

Theta0 = 2*L*max(eig(value(P)))/lambda; 
sat=[z1max,z2max,z3max,z4max,z5max,z6max];
%%
Theta =3;
T=[];
for i=1:n/2
T=[T Theta^i];
end
T=diag([T T]);
%%
%Simulation Observateur
%etat intial d'observateur
vertex={[-z1max,z1max],[-z2max,z2max],[-z3max,z3max],[-z4max,z4max],[-z5max,z5max],[-z6max,z6max]};
m0=10*ones(n,1);
%mesure y
%y=x(:,1:2)+0.1*normrnd(0,1.2,size(x(:,1:2)));
y=x(:,1:2);
y(randperm(length(y),350*0.4),1:2)=NaN;
%y(:,:)=NaN;
[t,m] = ode45(@(t,x) HighGainObserver_method2(t,x,A,B,C,T,K_,M_,y,v,vxy,Yaw_rate,tspan,vertex), tspan, m0);
[t,h] = ode45(@(t,x) HighGainObserver(t,x,A,B,C,T,K,y,v,tspan,vertex), tspan, m0);
%Calcul de l'angle PSi a partir des etat du system transformé
Psi = atan2(m(:,5),m(:,2));
Psi_HG= atan2(h(:,5),h(:,2));

%%
%Les figures
xlim([0,35]);
grid on
plot(tspan,x(:,3),'r',tspan,Psi,'k',tspan,Psi_HG,'g','LineWidth',4)

legend('$x_3$','$\hat{x}_3$ Multi-Output HG','$\hat{x}_3$ SHGO','Interpreter' ,'Latex')
ylabel("Yaw Rate (rad/s)",'LineWidth',18);
grid on
xlim([0,35]);
figure 
subplot(2,1,1)
plot(tspan,x(:,1),'r',tspan,m(:,1),'k--',tspan,h(:,1),'g--','LineWidth',4)
legend('$x_1$','$\hat{x}_1$ Multi-Output HG','$\hat{x}_1$ SHGO','Interpreter' ,'Latex')
subplot(2,1,2)
plot(tspan,x(:,2),'r',tspan,m(:,4),'k--',tspan,h(:,4),'g--','LineWidth',4)
legend('$x_2$','$\hat{x}_2$ Multi-Output HG','$\hat{x}_2$ SHGO','Interpreter' ,'Latex')
ylabel("Position (m)",'LineWidth',18);
grid on
xlim([0,35]);
figure 
hold on 
plot(x(:,1),x(:,2),'r','LineWidth',2)
plot(y(:,1),y(:,2),'co','LineWidth',2)
plot(m(:,1),m(:,4),'k','LineWidth',4)
plot(h(:,1),h(:,4),'g','LineWidth',4)
xlabel('X (m)','LineWidth',18)
ylabel('Y (m)','LineWidth',18)
legend("Ground Truth",'GPS measurement (observed output)','Multi-Output HG','SHGO')
grid on
% figure 
% 
% plot(tspan,abs(x(:,3)-Psi),'LineWidth',4)
% yticks(-2:0.2:2)
% grid on
% xlim([0,35]);
% figure
% subplot(2,1,1)
% plot(tspan,vxy(1,:),'r',tspan,m(:,2),'k--','LineWidth',4)
% grid on
% xlim([0,35]);
% legend('Estimated $V_x$','Measured $V_x$','Interpreter','Latex')
% subplot(2,1,2)
% plot(tspan,vxy(2,:),'r',tspan,m(:,5),'k--','LineWidth',4)
% grid on
% xlim([0,35]);
% legend('Estimated $V_y$','Measured $V_y$','Interpreter','Latex')
