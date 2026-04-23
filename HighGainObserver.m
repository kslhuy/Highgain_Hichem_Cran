function dy = HighGainObserver(t,x,A,B,C,T,K,y,v,tspan,varargin)
yc=interp1(tspan,y,t);
if isscalar(v)
    v_ = v;
else
    v_ = interp1(tspan,v,t);
end
if ~isnan(yc)
dy= A*x+B*Phi(x,v_)+T*K*(yc'-C*x);
else
dy= A*x+B*Phi(x,v_);
end
if ~isempty(varargin)
dy=Proj(dy,varargin{1});
end
end