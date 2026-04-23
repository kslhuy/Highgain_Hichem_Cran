function dy = LinkRobot(t,x,A,B,C,F,f,L)
dy(1:4,:)=A*x(1:4)+B*sin(2*pi*t)+F*feval(f,x(1:4));
dy(5:8,:)=A*x(5:8)+B*sin(2*pi*t)+F*feval(f,x(5:8))+L*C*(x(1:4)-x(5:8));
end