function dy = HighGainObserver_method1(t,x,A,B,C,T,K,M,y,v,tspan,varargin)
yc=interp1(tspan,y,t);
dy= A*x+B*Phi(x,v)+T*K*(yc'-C*x)+M*(v-sqrt(x(2)^2+x(5)^2));
if ~isempty(varargin)
dy=Proj(dy,varargin{1});
end
end

