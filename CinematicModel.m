function dy = CinematicModel(t,z,v,L,tspan)

%v_=interp1(tspan,v,t);
dy=[v*cos(z(3)), v*sin(z(3)),v/L*tan(z(4)),0]';
end

