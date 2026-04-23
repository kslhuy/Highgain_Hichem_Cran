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
tspan = 0:h:10;


%LMI
lambda = 1;
P= sdpvar(n/2,n/2) ; 
Y=sdpvar(1,n/2); 
M1= [ a'*P + P*a - c'*Y - Y'*c + lambda * eye(n/2), zeros(n/2,n/2);
    zeros(n/2,n/2), -P ];
const = [M1 <= 0];
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
z1max= 20;
z2max= 5;
z3max= 1;
z4max= z1max;
z5max= z2max;
z6max= z3max;
L=double(f(z1max, z2max,z3max,z4max, z5max, z6max));
%%
Theta0 = 2*L*max(eig(value(P)))/lambda; 
Theta = 3.7;
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
y=x(:,1:2)+0.1*rand(1001,2);
[t,m] = ode45(@(t,x) HighGainObserver(t,x,A,B,C,T,K,y,v,tspan), tspan, m0);
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
