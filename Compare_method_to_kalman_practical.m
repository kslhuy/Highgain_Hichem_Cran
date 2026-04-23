clearvars
close all
clc
rng(100)

% Practical comparison script for real vehicle deployment.
% Keeps EKF and adds a lightweight discrete observer with fixed gains.
load('Data.mat')

sim_window = 1:350;
tspan = time(sim_window);
tspan = tspan(:);
N = numel(tspan);

xy_true = position(:, sim_window)';
yaw_true = wrap_angle((yaw(sim_window) - yaw(1))');
vxy_raw = velocity(:, sim_window)';
yaw_rate_raw = yaw_rate(sim_window);
yaw_rate_raw = yaw_rate_raw(:);

cfg = struct();
cfg.gps_noise_std = 0.2;
cfg.gps_dropout_ratio = 0.20;
cfg.vel_noise_std = 0.05;
cfg.yaw_rate_noise_std = 0.03;
cfg.vel_lpf_window = 5;
cfg.yaw_lpf_window = 5;

cfg.ekf_speed_proc_std = 0.05;
cfg.ekf_yaw_rate_proc_std = 0.10;
cfg.omega_floor = 1e-3;

% Observer gains: tune these for your vehicle.
cfg.pos_gain = 0.35;
cfg.pos_to_vel_gain = 0.12;
cfg.pos_to_acc_gain = 0.03;
cfg.vel_gain = 0.30;
cfg.acc_from_vel_gain = 0.10;
cfg.yaw_correction_gain = 0.08;
cfg.min_speed_for_model = 0.2;
cfg.min_speed_for_heading = 0.5;

% Simulated sensors (replace this section with real sensors in deployment).
gps_meas = xy_true + cfg.gps_noise_std * randn(size(xy_true));
gps_dropout_mask = rand(N, 1) < cfg.gps_dropout_ratio;
gps_meas(gps_dropout_mask, :) = NaN;

vxy_meas = vxy_raw + cfg.vel_noise_std * randn(size(vxy_raw));
yaw_rate_meas = yaw_rate_raw + cfg.yaw_rate_noise_std * randn(size(yaw_rate_raw));

vxy_f = zeros(size(vxy_meas));
vxy_f(:, 1) = movmean(vxy_meas(:, 1), cfg.vel_lpf_window, 'omitnan');
vxy_f(:, 2) = movmean(vxy_meas(:, 2), cfg.vel_lpf_window, 'omitnan');
yaw_rate_f = movmean(yaw_rate_meas, cfg.yaw_lpf_window, 'omitnan');
speed_f = max(hypot(vxy_f(:, 1), vxy_f(:, 2)), cfg.min_speed_for_model);

first_gps_idx = find(all(~isnan(gps_meas), 2), 1, 'first');
if all(~isnan(gps_meas(1, :)))
    x0 = [gps_meas(1, 1); gps_meas(1, 2); yaw_true(1)];
elseif ~isempty(first_gps_idx)
    x0 = [gps_meas(first_gps_idx, 1); gps_meas(first_gps_idx, 2); yaw_true(1)];
else
    x0 = [xy_true(1, 1); xy_true(1, 2); yaw_true(1)];
end
P0 = diag([2^2, 2^2, (20 * pi / 180)^2]);

Q_gps = diag([cfg.gps_noise_std^2, cfg.gps_noise_std^2]);
R_motion = diag([cfg.ekf_speed_proc_std^2, cfg.ekf_speed_proc_std^2, cfg.ekf_yaw_rate_proc_std^2]);

kf = ExtendedKalmanFilter(x0, P0);
ekf = zeros(N, 3);
ekf(1, :) = x0';

for k = 2:N
    dt = max(tspan(k) - tspan(k - 1), 1e-3);
    omega = yaw_rate_f(k);

    if abs(omega) < cfg.omega_floor
        omega = cfg.omega_floor * sign(omega + eps);
    end

    u = [speed_f(k); omega];
    kf = kf.propagate(u, dt, R_motion * (dt^2));

    if all(~isnan(gps_meas(k, :)))
        kf = kf.update(gps_meas(k, :)', Q_gps);
    end

    kf.x(3) = wrap_angle(kf.x(3));
    ekf(k, :) = kf.x';
end

obs = zeros(N, 6);  % [px vx ax py vy ay]
obs_yaw = zeros(N, 1);
obs(1, :) = [x0(1), vxy_f(1, 1), 0, x0(2), vxy_f(1, 2), 0];
obs_yaw(1) = x0(3);

for k = 2:N
    dt = max(tspan(k) - tspan(k - 1), 1e-3);
    [obs(k, :), obs_yaw(k)] = practical_observer_step(...
        obs(k - 1, :), obs_yaw(k - 1), gps_meas(k, :), vxy_f(k, :), yaw_rate_f(k), dt, cfg);
end

err_pos_ekf = hypot(ekf(:, 1) - xy_true(:, 1), ekf(:, 2) - xy_true(:, 2));
err_pos_obs = hypot(obs(:, 1) - xy_true(:, 1), obs(:, 4) - xy_true(:, 2));
err_yaw_ekf = abs(wrap_angle(ekf(:, 3) - yaw_true));
err_yaw_obs = abs(wrap_angle(obs_yaw - yaw_true));

fprintf('RMS position error EKF: %.3f m\n', sqrt(mean(err_pos_ekf .^ 2))); 
fprintf('RMS position error Practical observer: %.3f m\n', sqrt(mean(err_pos_obs .^ 2))); 
fprintf('RMS yaw error EKF: %.3f rad\n', sqrt(mean(err_yaw_ekf .^ 2))); 
fprintf('RMS yaw error Practical observer: %.3f rad\n', sqrt(mean(err_yaw_obs .^ 2))); 

figure('Color', 'w')
subplot(2, 2, [1 3])
hold on
grid on
plot(xy_true(:, 1), xy_true(:, 2), 'k', 'LineWidth', 2)
plot(ekf(:, 1), ekf(:, 2), 'b', 'LineWidth', 2)
plot(obs(:, 1), obs(:, 4), 'r', 'LineWidth', 2)
idx_valid = all(~isnan(gps_meas), 2);
plot(gps_meas(idx_valid, 1), gps_meas(idx_valid, 2), '.', 'Color', [0 0.7 0.7])
legend('Ground truth', 'EKF', 'Practical observer', 'GPS (valid)', 'Location', 'best')
xlabel('X (m)')
ylabel('Y (m)')
title('Trajectory')

subplot(2, 2, 2)
hold on
grid on
plot(tspan, yaw_true, 'k', 'LineWidth', 1.5)
plot(tspan, ekf(:, 3), 'b', 'LineWidth', 1.5)
plot(tspan, obs_yaw, 'r', 'LineWidth', 1.5)
legend('Ground truth', 'EKF', 'Practical observer', 'Location', 'best')
xlabel('Time (s)')
ylabel('Yaw (rad)')
title('Yaw')

subplot(2, 2, 4)
hold on
grid on
plot(tspan, err_pos_ekf, 'b', 'LineWidth', 1.5)
plot(tspan, err_pos_obs, 'r', 'LineWidth', 1.5)
plot(tspan, err_yaw_ekf, 'b--', 'LineWidth', 1.2)
plot(tspan, err_yaw_obs, 'r--', 'LineWidth', 1.2)
legend('Pos err EKF', 'Pos err Practical', 'Yaw err EKF', 'Yaw err Practical', 'Location', 'best')
xlabel('Time (s)')
ylabel('Error')
title('Error overview')

function [x_next, yaw_next] = practical_observer_step(x_prev, yaw_prev, gps_xy, vxy_meas, yaw_rate_meas, dt, cfg)
[px, vx, ax] = axis_observer_update(x_prev(1), x_prev(2), x_prev(3), gps_xy(1), vxy_meas(1), dt, cfg);
[py, vy, ay] = axis_observer_update(x_prev(4), x_prev(5), x_prev(6), gps_xy(2), vxy_meas(2), dt, cfg);
x_next = [px, vx, ax, py, vy, ay];

if isnan(yaw_rate_meas)
    yaw_rate_meas = 0;
end

yaw_pred = wrap_angle(yaw_prev + dt * yaw_rate_meas);
speed_est = hypot(vx, vy);

if speed_est > cfg.min_speed_for_heading
    yaw_vel = atan2(vy, vx);
    yaw_innov = wrap_angle(yaw_vel - yaw_pred);
    yaw_next = wrap_angle(yaw_pred + cfg.yaw_correction_gain * yaw_innov);
else
    yaw_next = yaw_pred;
end
end

function [p_next, v_next, a_next] = axis_observer_update(p, v, a, p_meas, v_meas, dt, cfg)
p_pred = p + dt * v + 0.5 * dt^2 * a;
v_pred = v + dt * a;
a_pred = a;

if ~isnan(v_meas)
    v_innov = v_meas - v_pred;
    v_pred = v_pred + cfg.vel_gain * v_innov;
    a_pred = a_pred + cfg.acc_from_vel_gain * (v_innov / max(dt, 1e-3));
end

if ~isnan(p_meas)
    p_innov = p_meas - p_pred;
    p_next = p_pred + cfg.pos_gain * p_innov;
    v_next = v_pred + cfg.pos_to_vel_gain * (p_innov / max(dt, 1e-3));
    a_next = a_pred + cfg.pos_to_acc_gain * (2 * p_innov / max(dt^2, 1e-3));
else
    p_next = p_pred;
    v_next = v_pred;
    a_next = a_pred;
end
end

function angle = wrap_angle(angle)
angle = atan2(sin(angle), cos(angle));
end
