function dy = CinematicModelTransformed(t,z,v)

alpha= 1/v^2 * (-z(5)*z(3)+z(2)*z(6));
dy=[z(2), z(3),-alpha*z(6),z(5),z(6),alpha*z(3)]';
end

