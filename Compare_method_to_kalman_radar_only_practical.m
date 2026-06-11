clearvars
close all
clc
rng(100)

% Radar-only practical comparison.
% No camera is assumed. The estimator runs at the log/sample frequency, while
% radar measurements arrive at cfg.radar_rate_hz and can disappear randomly or
% in bursts. Replace simulate_radar_only_measurements with real radar logs in
% deployment.

load('Data.mat')

sim_window = 1:350;
tspan = time(sim_window);
tspan = tspan(:);
N = numel(tspan);

xy_true = position(:, sim_window)';
vxy_true = velocity(:, sim_window)';

cfg = struct();

% Data source.
% Use 'simulated' to test the algorithm with Data.mat ground truth.
% Use 'file' for a real radar log. See README_radar_only_tracking.md.
cfg.radar_source = 'simulated';
cfg.real_radar_file = 'RadarData.mat';
cfg.real_radar_time_var = 'radar_time';
cfg.real_radar_range_var = 'radar_range';
cfg.real_radar_bearing_var = 'radar_bearing';
cfg.real_radar_bearing_unit = 'rad'; % 'rad' or 'deg'
cfg.real_radar_range_rate_var = 'radar_range_rate';
cfg.real_radar_xy_var = 'radar_xy';
cfg.real_radar_vxy_var = 'radar_vxy';
cfg.real_radar_valid_var = 'radar_valid';

% Frequency settings.
cfg.radar_rate_hz = 12;
cfg.radar_stale_after_s = 2.5 / cfg.radar_rate_hz;

% Radar noise model: range [m], bearing [rad], radial speed [m/s].
cfg.radar_range_std = 0.20;
cfg.radar_bearing_std = 1.0 * pi / 180;
cfg.radar_range_rate_std = 0.08;
cfg.min_radar_range = 0.50;
cfg.radar_init_min_range_m = 5.0;

% Radar loss model for tests. Keep both random losses and burst losses to
% reproduce real driving cases where the target is briefly not detected.
cfg.radar_random_dropout_ratio = 0.15;
cfg.radar_num_burst_losses = 3;
cfg.radar_burst_duration_s = [0.7, 1.6];

% Practical high-gain observer tuning.
cfg.hg_pos_gain = 0.35;
cfg.hg_pos_to_vel_gain = 0.04;
cfg.hg_pos_to_acc_gain = 0.002;
cfg.hg_radial_vel_gain = 0.35;
cfg.hg_radial_vel_to_acc_gain = 0.02;
cfg.hg_accel_decay_time_s = 0.60;
cfg.hg_position_gate_m = 8.0;
cfg.hg_position_gate_bearing_sigma = 3.0;
cfg.hg_range_rate_gate_mps = 4.0;

% Radar-only EKF baseline tuning, using the same state as the observer:
% [px vx ax py vy ay]'.
cfg.ekf_process_accel_std = 1.4;
cfg.ekf_pos0_std = 4.0;
cfg.ekf_vel0_std = 4.0;
cfg.ekf_acc0_std = 3.0;
cfg.ekf_nis_gate = 25.0;

% Shared physical guards.
cfg.max_speed_mps = 25;
cfg.max_accel_mps2 = 10;

dt_log = median(diff(tspan));
log_rate_hz = 1 / max(dt_log, eps);
duration_s = max(tspan(end) - tspan(1), eps);

radar = get_radar_only_measurements(tspan, xy_true, vxy_true, cfg);

first_valid_idx = find(radar.valid & radar.range >= cfg.radar_init_min_range_m, 1, 'first');
if isempty(first_valid_idx)
    first_valid_idx = find(radar.valid, 1, 'first');
    if isempty(first_valid_idx)
        warning('No valid radar sample was generated. Falling back to first truth sample for initialization.')
        first_valid_idx = 1;
        x0 = [xy_true(1, 1); 0; 0; xy_true(1, 2); 0; 0];
    else
        warning('No radar sample reached cfg.radar_init_min_range_m. Initializing from the first valid radar sample.')
        x0 = initial_state_from_radar(radar, first_valid_idx);
    end
else
    x0 = initial_state_from_radar(radar, first_valid_idx);
end

P0 = diag([cfg.ekf_pos0_std^2, cfg.ekf_vel0_std^2, cfg.ekf_acc0_std^2, ...
           cfg.ekf_pos0_std^2, cfg.ekf_vel0_std^2, cfg.ekf_acc0_std^2]);

[ekf_state, ekf_update_used] = run_radar_only_ekf(tspan, radar, x0, P0, first_valid_idx, cfg);
[hg_state, hg_update_used, hg_pos_rejected, hg_rr_rejected] = run_radar_only_hg(tspan, radar, x0, first_valid_idx, cfg);

ekf_xy = [ekf_state(:, 1), ekf_state(:, 4)];
hg_xy = [hg_state(:, 1), hg_state(:, 4)];

err_pos_ekf = hypot(ekf_xy(:, 1) - xy_true(:, 1), ekf_xy(:, 2) - xy_true(:, 2));
err_pos_hg = hypot(hg_xy(:, 1) - xy_true(:, 1), hg_xy(:, 2) - xy_true(:, 2));

radar_gap_s = time_since_last_valid(tspan, radar.valid);
radar_lost_mask = radar_gap_s > cfg.radar_stale_after_s;
radar_fresh_mask = ~radar_lost_mask;
eval_mask = false(N, 1);
eval_mask(first_valid_idx:end) = true;

scheduled_rate_hz = nnz(radar.update_due) / duration_s;
valid_rate_hz = nnz(radar.valid) / duration_s;
scheduled_loss_ratio = nnz(radar.update_due & ~radar.valid) / max(nnz(radar.update_due), 1);
longest_gap_s = longest_valid_gap(tspan, radar.valid);

fprintf('Estimator/log frequency: %.2f Hz\n', log_rate_hz)
fprintf('Configured radar frequency: %.2f Hz\n', cfg.radar_rate_hz)
fprintf('Scheduled radar frequency on log: %.2f Hz\n', scheduled_rate_hz)
fprintf('Effective valid radar frequency: %.2f Hz\n', valid_rate_hz)
fprintf('Initialization time: %.3f s, radar range: %.3f m\n', tspan(first_valid_idx), radar.range(first_valid_idx))
fprintf('Scheduled radar loss ratio: %.1f %%\n', 100 * scheduled_loss_ratio)
fprintf('Radar stale threshold: %.3f s\n', cfg.radar_stale_after_s)
fprintf('Longest gap between valid radar updates: %.3f s\n', longest_gap_s)
fprintf('\n')
fprintf('RMS position error EKF radar-only: %.3f m\n', rms_masked(err_pos_ekf, eval_mask))
fprintf('RMS position error HG radar-only: %.3f m\n', rms_masked(err_pos_hg, eval_mask))
fprintf('RMS EKF while radar fresh: %.3f m\n', rms_masked(err_pos_ekf, eval_mask & radar_fresh_mask))
fprintf('RMS HG while radar fresh: %.3f m\n', rms_masked(err_pos_hg, eval_mask & radar_fresh_mask))
fprintf('RMS EKF while radar lost: %.3f m\n', rms_masked(err_pos_ekf, eval_mask & radar_lost_mask))
fprintf('RMS HG while radar lost: %.3f m\n', rms_masked(err_pos_hg, eval_mask & radar_lost_mask))
fprintf('Max HG error while radar lost: %.3f m\n', max_masked(err_pos_hg, eval_mask & radar_lost_mask))
fprintf('EKF accepted radar updates: %d / %d valid samples\n', nnz(ekf_update_used), nnz(radar.valid))
fprintf('HG accepted radar updates: %d / %d valid samples\n', nnz(hg_update_used), nnz(radar.valid))
fprintf('HG rejected position updates: %d\n', nnz(hg_pos_rejected))
fprintf('HG rejected radial-speed updates: %d\n', nnz(hg_rr_rejected))

figure('Color', 'w')
subplot(2, 2, [1 3])
hold on
grid on
axis equal
plot(xy_true(:, 1), xy_true(:, 2), 'k', 'LineWidth', 2)
plot(ekf_xy(:, 1), ekf_xy(:, 2), 'b', 'LineWidth', 1.8)
plot(hg_xy(:, 1), hg_xy(:, 2), 'r', 'LineWidth', 1.8)
plot(radar.xy(radar.valid, 1), radar.xy(radar.valid, 2), '.', 'Color', [0 0.6 0.6])
legend('Ground truth', 'Radar EKF', 'Radar-only HG', 'Valid radar', 'Location', 'best')
xlabel('X (m)')
ylabel('Y (m)')
title('Radar-only target tracking')

subplot(2, 2, 2)
hold on
grid on
plot(tspan, radar_gap_s, 'Color', [0.1 0.1 0.1], 'LineWidth', 1.5)
plot([tspan(1), tspan(end)], [cfg.radar_stale_after_s, cfg.radar_stale_after_s], 'r--', 'LineWidth', 1.2)
legend('Time since last valid radar', 'Stale threshold', 'Location', 'best')
xlabel('Time (s)')
ylabel('Gap (s)')
title('Radar availability')

subplot(2, 2, 4)
hold on
grid on
plot(tspan, err_pos_ekf, 'b', 'LineWidth', 1.5)
plot(tspan, err_pos_hg, 'r', 'LineWidth', 1.5)
plot(tspan(radar_lost_mask), err_pos_hg(radar_lost_mask), 'ko', 'MarkerSize', 3)
legend('EKF error', 'HG error', 'HG during radar loss', 'Location', 'best')
xlabel('Time (s)')
ylabel('Position error (m)')
title('Error during valid and lost radar periods')

figure('Color', 'w')
subplot(3, 1, 1)
hold on
grid on
plot(tspan, radar.range, '.', 'Color', [0 0.45 0.75])
ylabel('Range (m)')
title('Radar measurements')

subplot(3, 1, 2)
hold on
grid on
plot(tspan, radar.bearing, '.', 'Color', [0 0.45 0.75])
ylabel('Bearing (rad)')

subplot(3, 1, 3)
hold on
grid on
plot(tspan, radar.range_rate, '.', 'Color', [0 0.45 0.75])
plot(tspan(radar.update_due & ~radar.valid), zeros(nnz(radar.update_due & ~radar.valid), 1), 'rx')
xlabel('Time (s)')
ylabel('Range rate (m/s)')
legend('Valid radar value', 'Scheduled but lost', 'Location', 'best')

function radar = get_radar_only_measurements(tspan, xy_true, vxy_true, cfg)
switch lower(cfg.radar_source)
    case 'simulated'
        radar = simulate_radar_only_measurements(tspan, xy_true, vxy_true, cfg);
    case 'file'
        radar = load_radar_only_measurements(tspan, cfg);
    otherwise
        error('Unknown cfg.radar_source "%s". Use "simulated" or "file".', cfg.radar_source)
end
end

function radar = simulate_radar_only_measurements(tspan, xy_true, vxy_true, cfg)
N = numel(tspan);
duration_s = max(tspan(end) - tspan(1), eps);
radar_period_s = 1 / cfg.radar_rate_hz;

update_due = false(N, 1);
next_update_time = tspan(1);
for k = 1:N
    if tspan(k) + 10 * eps >= next_update_time
        update_due(k) = true;
        next_update_time = next_update_time + radar_period_s;
    end
end

range_true = hypot(xy_true(:, 1), xy_true(:, 2));
range_safe = max(range_true, cfg.min_radar_range);
bearing_true = atan2(xy_true(:, 2), xy_true(:, 1));
range_rate_true = (xy_true(:, 1) .* vxy_true(:, 1) + xy_true(:, 2) .* vxy_true(:, 2)) ./ range_safe;

random_loss = rand(N, 1) < cfg.radar_random_dropout_ratio;
burst_loss = false(N, 1);
if cfg.radar_num_burst_losses > 0 && duration_s > cfg.radar_burst_duration_s(2)
    for i = 1:cfg.radar_num_burst_losses
        burst_duration = cfg.radar_burst_duration_s(1) + ...
            diff(cfg.radar_burst_duration_s) * rand(1);
        latest_start = max(tspan(1), tspan(end) - burst_duration);
        start_time = tspan(1) + rand(1) * max(latest_start - tspan(1), eps);
        burst_loss = burst_loss | (tspan >= start_time & tspan <= start_time + burst_duration);
    end
end

valid = update_due & ~random_loss & ~burst_loss;

range = range_true + cfg.radar_range_std * randn(N, 1);
bearing = wrap_angle(bearing_true + cfg.radar_bearing_std * randn(N, 1));
range_rate = range_rate_true + cfg.radar_range_rate_std * randn(N, 1);

range(~valid) = NaN;
bearing(~valid) = NaN;
range_rate(~valid) = NaN;

xy = NaN(N, 2);
xy(valid, 1) = range(valid) .* cos(bearing(valid));
xy(valid, 2) = range(valid) .* sin(bearing(valid));

radar = struct();
radar.update_due = update_due;
radar.valid = valid;
radar.range = range;
radar.bearing = bearing;
radar.range_rate = range_rate;
radar.xy = xy;
end

function radar = load_radar_only_measurements(tspan, cfg)
if ~isfile(cfg.real_radar_file)
    error('Radar file "%s" not found. Set cfg.real_radar_file or use cfg.radar_source = "simulated".', cfg.real_radar_file)
end

raw = load(cfg.real_radar_file);
radar_time = get_required_var(raw, cfg.real_radar_time_var);
radar_time = radar_time(:);

has_polar = isfield(raw, cfg.real_radar_range_var) && ...
    isfield(raw, cfg.real_radar_bearing_var) && ...
    isfield(raw, cfg.real_radar_range_rate_var);
has_xy = isfield(raw, cfg.real_radar_xy_var);

if has_polar
    radar_range = get_required_var(raw, cfg.real_radar_range_var);
    radar_bearing = get_required_var(raw, cfg.real_radar_bearing_var);
    radar_range_rate = get_required_var(raw, cfg.real_radar_range_rate_var);
    radar_range = radar_range(:);
    radar_bearing = radar_bearing(:);
    radar_range_rate = radar_range_rate(:);

    if strcmpi(cfg.real_radar_bearing_unit, 'deg')
        radar_bearing = deg2rad(radar_bearing);
    end

    radar_xy = [radar_range .* cos(radar_bearing), radar_range .* sin(radar_bearing)];
elseif has_xy
    radar_xy = get_required_var(raw, cfg.real_radar_xy_var);
    if size(radar_xy, 2) ~= 2
        error('Variable "%s" must be an Nx2 matrix [x y].', cfg.real_radar_xy_var)
    end

    radar_range = hypot(radar_xy(:, 1), radar_xy(:, 2));
    radar_bearing = atan2(radar_xy(:, 2), radar_xy(:, 1));

    if isfield(raw, cfg.real_radar_range_rate_var)
        radar_range_rate = raw.(cfg.real_radar_range_rate_var);
        radar_range_rate = radar_range_rate(:);
    elseif isfield(raw, cfg.real_radar_vxy_var)
        radar_vxy = raw.(cfg.real_radar_vxy_var);
        if size(radar_vxy, 2) ~= 2
            error('Variable "%s" must be an Nx2 matrix [vx vy].', cfg.real_radar_vxy_var)
        end
        radar_range_safe = max(radar_range, cfg.min_radar_range);
        radar_range_rate = (radar_xy(:, 1) .* radar_vxy(:, 1) + ...
            radar_xy(:, 2) .* radar_vxy(:, 2)) ./ radar_range_safe;
    else
        radar_range_rate = estimate_range_rate(radar_time, radar_range);
    end
else
    error(['Radar file must contain either polar variables "%s", "%s", "%s" ', ...
           'or Cartesian variable "%s".'], ...
           cfg.real_radar_range_var, cfg.real_radar_bearing_var, ...
           cfg.real_radar_range_rate_var, cfg.real_radar_xy_var)
end

if isfield(raw, cfg.real_radar_valid_var)
    radar_valid = logical(raw.(cfg.real_radar_valid_var));
    radar_valid = radar_valid(:);
else
    radar_valid = true(size(radar_time));
end

check_radar_lengths(radar_time, radar_range, radar_bearing, radar_range_rate, radar_valid)

log_dt = median(diff(tspan));
time_margin = 0.5 * max(log_dt, eps);
in_log_window = radar_time >= tspan(1) - time_margin & ...
    radar_time <= tspan(end) + time_margin;
radar_time = radar_time(in_log_window);
radar_range = radar_range(in_log_window);
radar_bearing = radar_bearing(in_log_window);
radar_range_rate = radar_range_rate(in_log_window);
radar_xy = radar_xy(in_log_window, :);
radar_valid = radar_valid(in_log_window);

if isempty(radar_time)
    error('No radar samples overlap the selected tspan window.')
end

radar_valid = radar_valid & isfinite(radar_time) & isfinite(radar_range) & ...
    isfinite(radar_bearing) & isfinite(radar_range_rate) & radar_range >= cfg.min_radar_range;

radar = map_radar_samples_to_log(tspan, radar_time, radar_range, ...
    radar_bearing, radar_range_rate, radar_xy, radar_valid);
end

function value = get_required_var(raw, var_name)
if ~isfield(raw, var_name)
    error('Required variable "%s" is missing from the radar file.', var_name)
end
value = raw.(var_name);
end

function check_radar_lengths(radar_time, radar_range, radar_bearing, radar_range_rate, radar_valid)
n = numel(radar_time);
if any([numel(radar_range), numel(radar_bearing), numel(radar_range_rate), numel(radar_valid)] ~= n)
    error('Radar variables must all have the same number of samples.')
end
end

function range_rate = estimate_range_rate(radar_time, radar_range)
range_rate = zeros(size(radar_range));
if numel(radar_range) < 2
    return
end

dt = diff(radar_time);
dr = diff(radar_range);
valid_dt = abs(dt) > eps;
instant_rate = zeros(size(dr));
instant_rate(valid_dt) = dr(valid_dt) ./ dt(valid_dt);
range_rate(1) = instant_rate(1);
range_rate(2:end) = instant_rate;
end

function radar = map_radar_samples_to_log(tspan, radar_time, radar_range, radar_bearing, radar_range_rate, radar_xy, radar_valid)
N = numel(tspan);
radar = struct();
radar.update_due = false(N, 1);
radar.valid = false(N, 1);
radar.range = NaN(N, 1);
radar.bearing = NaN(N, 1);
radar.range_rate = NaN(N, 1);
radar.xy = NaN(N, 2);

for i = 1:numel(radar_time)
    [~, idx] = min(abs(tspan - radar_time(i)));
    radar.update_due(idx) = true;

    if radar_valid(i)
        radar.valid(idx) = true;
        radar.range(idx) = radar_range(i);
        radar.bearing(idx) = wrap_angle(radar_bearing(i));
        radar.range_rate(idx) = radar_range_rate(i);
        radar.xy(idx, :) = radar_xy(i, :);
    end
end
end

function x0 = initial_state_from_radar(radar, idx)
los = [cos(radar.bearing(idx)); sin(radar.bearing(idx))];
pos = radar.range(idx) * los;
vel = radar.range_rate(idx) * los;
x0 = [pos(1); vel(1); 0; pos(2); vel(2); 0];
end

function [state, update_used] = run_radar_only_ekf(tspan, radar, x0, P0, first_valid_idx, cfg)
N = numel(tspan);
state = repmat(x0', N, 1);
update_used = false(N, 1);

x = x0;
P = P0;
state(first_valid_idx, :) = x';

for k = max(first_valid_idx + 1, 2):N
    dt = max(tspan(k) - tspan(k - 1), 1e-3);
    [x, P] = ca_ekf_predict(x, P, dt, cfg);

    if radar.valid(k)
        z = [radar.range(k); radar.bearing(k); radar.range_rate(k)];
        [x, P, update_used(k)] = radar_ekf_update(x, P, z, cfg);
    end

    x = limit_state_norms(x, cfg);
    state(k, :) = x';
end
end

function [x, P] = ca_ekf_predict(x, P, dt, cfg)
F_axis = [1, dt, 0.5 * dt^2;
          0, 1, dt;
          0, 0, 1];
F = blkdiag(F_axis, F_axis);

G_axis = [0.5 * dt^2; dt; 1];
Q_axis = cfg.ekf_process_accel_std^2 * (G_axis * G_axis');
Q = blkdiag(Q_axis, Q_axis);

x = F * x;
P = F * P * F' + Q;
P = (P + P') / 2;
end

function [x, P, accepted] = radar_ekf_update(x, P, z, cfg)
accepted = false;

px = x(1);
vx = x(2);
py = x(4);
vy = x(5);

r = max(hypot(px, py), cfg.min_radar_range);
r2 = r^2;
r3 = r^3;
dot_pr = px * vx + py * vy;

h = [r;
     atan2(py, px);
     dot_pr / r];

H = zeros(3, 6);
H(1, 1) = px / r;
H(1, 4) = py / r;
H(2, 1) = -py / r2;
H(2, 4) = px / r2;
H(3, 1) = vx / r - px * dot_pr / r3;
H(3, 2) = px / r;
H(3, 4) = vy / r - py * dot_pr / r3;
H(3, 5) = py / r;

R = diag([cfg.radar_range_std^2, cfg.radar_bearing_std^2, cfg.radar_range_rate_std^2]);
innovation = z - h;
innovation(2) = wrap_angle(innovation(2));

S = H * P * H' + R;
nis = innovation' / S * innovation;
if nis > cfg.ekf_nis_gate
    return
end

K = P * H' / S;
x = x + K * innovation;
I = eye(size(P));
P = (I - K * H) * P * (I - K * H)' + K * R * K';
P = (P + P') / 2;
accepted = true;
end

function [state, update_used, pos_rejected, rr_rejected] = run_radar_only_hg(tspan, radar, x0, first_valid_idx, cfg)
N = numel(tspan);
state = repmat(x0', N, 1);
update_used = false(N, 1);
pos_rejected = false(N, 1);
rr_rejected = false(N, 1);

x = x0;
state(first_valid_idx, :) = x';
last_valid_time = tspan(first_valid_idx);

for k = max(first_valid_idx + 1, 2):N
    dt = max(tspan(k) - tspan(k - 1), 1e-3);
    if isnan(last_valid_time)
        gap_s = inf;
    else
        gap_s = tspan(k) - last_valid_time;
    end

    x = hg_predict(x, dt, gap_s, cfg);

    if radar.valid(k)
        [x, pos_ok, rr_ok] = hg_radar_update(x, radar, k, dt, cfg);
        update_used(k) = pos_ok || rr_ok;
        pos_rejected(k) = ~pos_ok;
        rr_rejected(k) = ~rr_ok;
        if update_used(k)
            last_valid_time = tspan(k);
        end
    end

    x = limit_state_norms(x, cfg);
    state(k, :) = x';
end
end

function x = hg_predict(x, dt, gap_s, cfg)
if gap_s > cfg.radar_stale_after_s
    decay = exp(-dt / cfg.hg_accel_decay_time_s);
    x(3) = decay * x(3);
    x(6) = decay * x(6);
end

x(1) = x(1) + dt * x(2) + 0.5 * dt^2 * x(3);
x(2) = x(2) + dt * x(3);
x(4) = x(4) + dt * x(5) + 0.5 * dt^2 * x(6);
x(5) = x(5) + dt * x(6);
end

function [x, pos_ok, rr_ok] = hg_radar_update(x, radar, idx, dt, cfg)
pos_ok = false;
rr_ok = false;

range = radar.range(idx);
bearing = radar.bearing(idx);
range_rate = radar.range_rate(idx);

los = [cos(bearing); sin(bearing)];
pos_meas = range * los;
pos_pred = [x(1); x(4)];
pos_innov = pos_meas - pos_pred;

position_gate = cfg.hg_position_gate_m + ...
    cfg.hg_position_gate_bearing_sigma * range * cfg.radar_bearing_std;

if norm(pos_innov) <= position_gate
    x(1) = x(1) + cfg.hg_pos_gain * pos_innov(1);
    x(4) = x(4) + cfg.hg_pos_gain * pos_innov(2);
    x(2) = x(2) + cfg.hg_pos_to_vel_gain * pos_innov(1) / max(dt, 1e-3);
    x(5) = x(5) + cfg.hg_pos_to_vel_gain * pos_innov(2) / max(dt, 1e-3);
    x(3) = x(3) + cfg.hg_pos_to_acc_gain * 2 * pos_innov(1) / max(dt^2, 1e-3);
    x(6) = x(6) + cfg.hg_pos_to_acc_gain * 2 * pos_innov(2) / max(dt^2, 1e-3);
    pos_ok = true;
end

vel_pred = [x(2); x(5)];
range_rate_pred = los' * vel_pred;
range_rate_innov = range_rate - range_rate_pred;

if abs(range_rate_innov) <= cfg.hg_range_rate_gate_mps
    dv = cfg.hg_radial_vel_gain * range_rate_innov * los;
    da = cfg.hg_radial_vel_to_acc_gain * range_rate_innov / max(dt, 1e-3) * los;
    x(2) = x(2) + dv(1);
    x(5) = x(5) + dv(2);
    x(3) = x(3) + da(1);
    x(6) = x(6) + da(2);
    rr_ok = true;
end
end

function x = limit_state_norms(x, cfg)
v = [x(2); x(5)];
speed = norm(v);
if speed > cfg.max_speed_mps
    v = v * (cfg.max_speed_mps / speed);
    x(2) = v(1);
    x(5) = v(2);
end

a = [x(3); x(6)];
accel = norm(a);
if accel > cfg.max_accel_mps2
    a = a * (cfg.max_accel_mps2 / accel);
    x(3) = a(1);
    x(6) = a(2);
end
end

function gap_s = time_since_last_valid(tspan, valid)
N = numel(tspan);
gap_s = inf(N, 1);
last_time = NaN;
for k = 1:N
    if valid(k)
        last_time = tspan(k);
    end
    if ~isnan(last_time)
        gap_s(k) = tspan(k) - last_time;
    end
end
end

function gap_s = longest_valid_gap(tspan, valid)
valid_times = tspan(valid);
if numel(valid_times) < 2
    gap_s = NaN;
else
    gap_s = max(diff(valid_times));
end
end

function value = rms_masked(x, mask)
if any(mask)
    value = sqrt(mean(x(mask) .^ 2));
else
    value = NaN;
end
end

function value = max_masked(x, mask)
if any(mask)
    value = max(x(mask));
else
    value = NaN;
end
end

function angle = wrap_angle(angle)
angle = atan2(sin(angle), cos(angle));
end
