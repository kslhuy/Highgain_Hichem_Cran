clear all 
clc
lambda=3.33;
A_=[0 1 0 0; -48.6 -1.25 48.6 0; 0 0 0 1 ; 19.5 0 -19.5  0];
A{1} = [0 1 0 0; -48.6 -1.25 48.6 0; 0 0 0 1 ; 19.5 0 -19.5-lambda  0];
A{2}= [0 1 0 0; -48.6 -1.25 48.6 0; 0 0 0 1 ; 19.5 0 -19.5+lambda  0];

C= [ eye(2) zeros(2,2)];
f=@(z) -3.33*sin(z(3));
F=[0 0 0 1]'; 
B=[0 21.6  0 0]';
na=size(A_,1);
nc=size(C,1);
P= sdpvar(na,na) ; 
Y=sdpvar(nc,na); 
const=[P >= 0];
for i=1:length(A)
const = [const; A{i}'*P + P*A{i} - C'*Y - Y'*C <= 0];
end
options = sdpsettings('solver','sdpt3');
diagnostic=optimize(const,[],options);
clc
display(diagnostic.info)
if diagnostic.problem== 0
P= value(P);
Y= value(Y);
L=P\Y'
end
%L=[4.0234 -2.1900;1.9856 40.5467;5.0345 12.7832;4.5506 -13.8713];
h=.001;
tspan=0:h:10;
x0=[[0 0 1 1]';zeros(4,1)];
[t,z] = ode45(@(t,x) LinkRobot(t,x,A_,B,C,F,f,L) , tspan, x0);
x=z(:,1:4);
x_hat=z(:,5:8);
i=3;
plot(t,x(:,i),'r',t,x_hat(:,i),'k--','LineWidth',2)
figure 
plot(t,x(:,i)-x_hat(:,i),'LineWidth',2)