clear all 
close all
clc

%System 
L_f=3;
v=4;
n= 6;
a=[zeros(2,1) eye(2) ;zeros(1,n/2)];
A=blkdiag(a,a);
c=[1 zeros(1,n/2-1)];
C= blkdiag(c,c);
b=[zeros(n/2-1,1); 1];
B=blkdiag(b,b);
%Simulation
h=0.01;
tspan = 0:h:2;


%LMI
lambda = 1;
P= sdpvar(n/2,n/2) ; 
Y=sdpvar(1,n/2); 
M1= [ a'*P + P*a - c'*Y - Y'*c + lambda * eye(n/2), zeros(n/2,n/2);
    zeros(n/2,n/2), -P ];
const = [M1 <= 0;];
diagnostic=optimize(const);
clc
K=value(P)\value(Y)';
K=[K zeros(n/2,1);zeros(n/2,1) K];

%Simulation du system transformé 
%z0=1*ones(n,1);
%[t,z] = ode45(@(t,z) CinematicModelTransformed(t,z,v), tspan, z0);

%Simulation du system 
%etat initial
x0=1*ones(4,1);
x0(3)=1;
x0(4)=.3;
%u=[0.3*ones(round(length(tspan)/2),1); zeros(floor(length(tspan)/2),1)];
%u=0.3*sin(tspan);
[t,x] = ode45(@(t,z) CinematicModel(t,z,v,L_f,tspan), tspan, x0);
%Transformation du l'angle Psi de R a [-pi,pi]
x(:,3)=NormlizeAngle(x(:,3));
%Observateur Grand gain 
%calcul de thetha 
syms z1 z2 z3 z4 z5 z6 
f1= -1/(z2^2+z5^2)*(-z5*z3+z2*z6)*z6;
f2= 1/(z2^2+z5^2)*(-z5*z3+z2*z6)*z3;
J=jacobian([f1,f2],[z1, z2, z3, z4, z5, z6 ]);
syms f(z1, z2, z3, z4, z5, z6)
f(z1, z2, z3, z4, z5, z6)=norm(J);
z1max= 40;
z2max= 11;
z3max= 6;
z4max= z1max;
z5max= z2max;
z6max= z3max;
vertex={[-z1max,z1max],[-z2max,z2max],[-z3max,z3max],[-z3max,z3max],[-z3max,z3max],[-z3max,z3max]};
L=double(f(z1max, z2max,z3max,z4max, z5max, z6max));

%% Information supplémentaire 
%les fonction nonlineaire 
h1=sqrt(z2^2+z5^2);
%Calcule du gradient
Jh=jacobian(h1,[z1, z2, z3, z4, z5, z6 ]);
syms h(z1, z2, z3, z4, z5, z6)
h(z1, z2, z3, z4, z5, z6)=norm(Jh);
%cte de Liptz
k_h=double(h(z1max, z2max,z3max,z4max, z5max, z6max));
%Choix de ||M||

alpha=0.1;
%%Calcule borne de theta
Theta01 = 2*L*max(eig(value(P)))/(lambda*alpha); 
Theta = (Theta01+0.2);
M_norm=lambda*(1-alpha)/(2*k_h*Theta*max(eig(value(P))));
Theta02 = lambda*(1-alpha)/(2*k_h*M_norm*max(eig(value(P))));

M=-M_norm*ones(n,1);
Theta=3.5;
T=[];
for i=1:n/2
T=[T Theta^i];
end
T=diag([T T]);
%%
%Simulation Observateur
%etat intial d'observateur
m0=5*ones(n,1);
%mesure y
%y=x(:,1:2)+0.2*rand(size(x(:,1:2)));
y=x(:,1:2);
[t,m] = ode45(@(t,x) HighGainObserver_method1(t,x,A,B,C,T,K,M,y,v,tspan,vertex), tspan, m0);
%Calcul de l'angle PSi a partir des etat du system transformé
Psi = atan2(m(:,5),m(:,2));
%Les figures
plot(t,x(:,3),'r',t,Psi,'k--','LineWidth',2)
legend('$x_3$','$\hat{x}_3$','Interpreter' ,'Latex')
figure 
plot(t,x(:,1),t,m(:,1),'--',t,x(:,2),t,m(:,4),'--','LineWidth',2)
legend('$x_1$','$\hat{x}_1$','$x_2$','$\hat{x}_2$','Interpreter' ,'Latex')
figure 
plot(x(:,1),x(:,2),'r',m(:,1),m(:,4),'k--','LineWidth',2)
xlabel('X (m)')
ylabel('Y (m)')
figure 
plot(t,abs(x(:,3)-Psi),'LineWidth',2)
%%
%Comparison side to side 
%Calcul de l'angle PSi a partir des etat du system transformé
close all
z=load('Meth2.mat');
z=z.m;
Psi1 = atan2(m(:,5),m(:,2));
Psi2=atan2(z(:,5),z(:,2));
% Method 2 data can have a different sampling length than t
t2 = linspace(t(1), t(end), size(z,1))';
if numel(t2) ~= numel(t)
    Psi2_on_t = interp1(t2, Psi2, t, 'linear', 'extrap');
else
    Psi2_on_t = Psi2;
end
%Les figures
subplot(2,2,[3,4])
plot(t,x(:,3),'k',t,Psi1,'r--',t2,Psi2,'b--','LineWidth',2)
legend('$x_3$','$\hat{x}_{31}$','$\hat{x}_{32}$','Interpreter' ,'Latex')
 subplot(2,2,1)
plot(t,x(:,1),'k',t,m(:,1),'r--',t2,z(:,1),'b--','LineWidth',2)
legend('$x_1$','$\hat{x}_{11}$','$\hat{x}_{12}$','Interpreter' ,'Latex')
 subplot(2,2,2)
plot(t,x(:,2),'k',t,m(:,4),'r--',t2,z(:,4),'b--','LineWidth',2)
legend('$x_2$','$\hat{x}_{21}$','$\hat{x}_{22}$','Interpreter' ,'Latex')
figure 
plot(t,abs(x(:,3)-Psi1),'r',t,abs(x(:,3)-Psi2_on_t),'b','LineWidth',2)
legend('Method 1 $|x_3-\hat{x}_{31}|$','Method 2 $|x_3-\hat{x}_{32}|$','Interpreter' ,'Latex')