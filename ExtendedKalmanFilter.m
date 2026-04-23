classdef ExtendedKalmanFilter
    properties
        x  % state to estimate: [x_, y_, theta]^T
        P  % estimation error covariance
    end
    methods
        function obj = ExtendedKalmanFilter(x, P)
            obj.x = x;
            obj.P = P;
        end
        function obj = update(obj, z, Q)
            % compute Kalman gain
            H = [1, 0, 0; 0, 1, 0];  % Jacobian of observation function
            K = obj.P * H' / (H * obj.P * H' + Q);

            % update state x
            x = obj.x(1); y = obj.x(2); theta = obj.x(3);
            z_ = [x; y];  % expected observation from the estimated state
            obj.x = obj.x + K * (z - z_);

            % update covariance P
            obj.P = obj.P - K * H * obj.P;
        end
        function obj = propagate(obj, u, dt, R)
            % propagate state x
            x = obj.x(1); y = obj.x(2); theta = obj.x(3);
            v = u(1); omega = u(2);
            r = v / omega;  % turning radius

            dtheta = omega * dt;
            dx = - r * sin(theta) + r * sin(theta + dtheta);
            dy = + r * cos(theta) - r * cos(theta + dtheta);

            obj.x = obj.x + [dx; dy; dtheta];

            % propagate covariance P
            G = [1, 0, - r * cos(theta) + r * cos(theta + dtheta);
                 0, 1, - r * sin(theta) + r * sin(theta + dtheta);
                 0, 0, 1];  % Jacobian of state transition function

            obj.P = G * obj.P * G' + R;
        end
    end
end