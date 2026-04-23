function dy = HighGainObserver_method2(t,x,A,B,C,T,K,M,y,v,vxy,yawr,tspan,varargin)
yc=interp1(tspan,y,t);
if isscalar(v)
    v_ = v;
else
    v_ = interp1(tspan,v,t);
end
Yawrate=interp1(tspan,yawr,t);
vx=interp1(tspan,vxy(1,:),t);
vy=interp1(tspan,vxy(2,:),t);
if ~isnan(yc)
%dy= A*x+B*Phi(x,v_)+T*K*(yc'-C*x)+T*M*[vx-x(2); vy-x(5)]+T*M*[Yawrate-((-x(5)*x(3)+x(2)*x(6))/(x(2)^2+x(5)^2)); Yawrate-((-x(5)*x(3)+x(2)*x(6))/(x(2)^2+x(5)^2))];
dy= A*x+B*Phi(x,v_)+T*K*(yc'-C*x)+T*M*[vx-x(2); vy-x(5)];
else
%dy= A*x+B*Phi(x,v_)+T*M*[vx-x(2); vy-x(5)]+T*M*[Yawrate-((-x(5)*x(3)+x(2)*x(6))/(x(2)^2+x(5)^2)); Yawrate-((-x(5)*x(3)+x(2)*x(6))/(x(2)^2+x(5)^2))];
dy= A*x+B*Phi(x,v_)+T*M*[vx-x(2); vy-x(5)];
end
if ~isempty(varargin)
dy=Proj(dy,varargin{1});
end
t
end