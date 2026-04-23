function dy = CinematicModelObs(t,z,v,L,y,C,K,tspan)

y_c=interp1(tspan,y,t)';
dy=[v*cos(z(3)); v*sin(z(3));v/L*tan(z(4));0]+K*(y_c-C*z);
t
end

