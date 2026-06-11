# Radar-only target vehicle tracking

This note explains the practical script:

```matlab
Compare_method_to_kalman_radar_only_practical.m
```

The problem is target vehicle tracking without camera. The only perception
source is radar, and the radar can lose the target for short periods.

The objective is to estimate the target vehicle state:

```text
[px, vx, ax, py, vy, ay]
```

where:

```text
px, py = 2D target position
vx, vy = 2D target velocity
ax, ay = 2D target acceleration
```

The estimator must keep predicting the target position when radar detections
are missing.

## What the script does

The script compares two radar-only estimators:

```text
1. Radar-only EKF baseline
2. Practical high-gain-style observer
```

Both use the same constant-acceleration state. The high-gain observer is tuned
to correct quickly when radar is available, then coast with velocity and
acceleration when radar is lost.

The default mode is simulated radar, generated from `Data.mat`:

```matlab
cfg.radar_source = 'simulated';
```

This lets you test frequency, noise, and dropout behavior before connecting
real radar logs.

## Radar measurement model

The preferred radar input is polar:

```text
range       = distance to target [m]
bearing     = azimuth angle to target [rad] or [deg]
range_rate  = radial velocity [m/s]
valid       = 1 when radar sees the target, 0 when lost
```

The script converts this to Cartesian position:

```matlab
x = range * cos(bearing)
y = range * sin(bearing)
```

The radial velocity corrects only the velocity component along the radar line
of sight.

Important limitation: if the radar gives only `range` without `bearing`, full
2D tracking is not observable. You need bearing, lateral position, or an
object track already computed by the radar.

## Frequency handling

There are two different frequencies:

```text
Estimator/log frequency
Radar detection frequency
```

In the current `Data.mat` test, the log frequency is about 10 Hz. The radar is
configured separately:

```matlab
cfg.radar_rate_hz = 12;
```

Because the log is 10 Hz, a simulated radar configured at 12 Hz is effectively
sampled on the 10 Hz log grid. The script prints both:

```text
Estimator/log frequency
Configured radar frequency
Scheduled radar frequency on log
Effective valid radar frequency
```

A radar sample is considered stale after:

```matlab
cfg.radar_stale_after_s = 2.5 / cfg.radar_rate_hz;
```

After this delay, the observer knows it is no longer receiving fresh radar
updates. It keeps predicting, but acceleration is slowly damped to avoid
unbounded drift.

## Radar loss model in simulation

The simulated radar has two loss modes:

```matlab
cfg.radar_random_dropout_ratio = 0.15;
cfg.radar_num_burst_losses = 3;
cfg.radar_burst_duration_s = [0.7, 1.6];
```

This means:

```text
15 percent of scheduled detections are randomly lost
3 longer radar-loss windows are inserted
each long loss lasts between 0.7 s and 1.6 s
```

These settings are only for stress testing. In real deployment, the radar log
provides the actual `valid` flag.

## Using a real radar log

Change the source:

```matlab
cfg.radar_source = 'file';
cfg.real_radar_file = 'RadarData.mat';
```

The `.mat` file should contain a radar timestamp vector:

```matlab
radar_time
```

Then it can use either polar radar variables:

```matlab
radar_range
radar_bearing
radar_range_rate
radar_valid
```

or Cartesian radar variables:

```matlab
radar_xy       % Nx2 matrix [x y]
radar_vxy      % optional Nx2 matrix [vx vy]
radar_valid
```

The variable names are configurable:

```matlab
cfg.real_radar_time_var = 'radar_time';
cfg.real_radar_range_var = 'radar_range';
cfg.real_radar_bearing_var = 'radar_bearing';
cfg.real_radar_bearing_unit = 'rad'; % 'rad' or 'deg'
cfg.real_radar_range_rate_var = 'radar_range_rate';
cfg.real_radar_xy_var = 'radar_xy';
cfg.real_radar_vxy_var = 'radar_vxy';
cfg.real_radar_valid_var = 'radar_valid';
```

If `radar_xy` is provided but no `radar_range_rate` or `radar_vxy` is present,
the script estimates range rate from range differences. This is less reliable
than the radar's native radial speed.

## Initialization

Radar angle is unstable when the target is extremely close to the radar. The
script waits until the first valid radar point with sufficient range:

```matlab
cfg.radar_init_min_range_m = 5.0;
```

Then it initializes:

```matlab
position = range * [cos(bearing), sin(bearing)]
velocity = range_rate * [cos(bearing), sin(bearing)]
acceleration = 0
```

The tangential velocity is initially unknown in a radar-only setup. The
estimator learns it through later position and bearing changes.

## Main high-gain observer settings

Position correction:

```matlab
cfg.hg_pos_gain = 0.35;
cfg.hg_pos_to_vel_gain = 0.04;
cfg.hg_pos_to_acc_gain = 0.002;
```

Radial-speed correction:

```matlab
cfg.hg_radial_vel_gain = 0.35;
cfg.hg_radial_vel_to_acc_gain = 0.02;
```

Outlier rejection:

```matlab
cfg.hg_position_gate_m = 8.0;
cfg.hg_position_gate_bearing_sigma = 3.0;
cfg.hg_range_rate_gate_mps = 4.0;
```

Physical limits:

```matlab
cfg.max_speed_mps = 25;
cfg.max_accel_mps2 = 10;
```

Acceleration damping during radar loss:

```matlab
cfg.hg_accel_decay_time_s = 0.60;
```

If the estimate reacts too much to radar noise, reduce `hg_pos_gain`,
`hg_pos_to_vel_gain`, and `hg_radial_vel_gain`. If it lags behind the target,
increase them carefully.

## Metrics printed by the script

The script prints:

```text
Estimator/log frequency
Configured radar frequency
Effective valid radar frequency
Radar loss ratio
Longest gap between valid radar updates
RMS error while radar is fresh
RMS error while radar is lost
Max error while radar is lost
Accepted/rejected radar updates
```

The most important demo metric is:

```text
RMS HG while radar lost
```

This shows how well the estimator keeps tracking the target during radar
dropouts.

## Current result on Data.mat

With the default simulated radar losses, the script produced:

```text
Estimator/log frequency: 10.00 Hz
Configured radar frequency: 12.00 Hz
Effective valid radar frequency: 7.83 Hz
Scheduled radar loss ratio: 19.1 %
Longest gap between valid radar updates: 1.560 s

RMS position error EKF radar-only: 1.895 m
RMS position error HG radar-only: 1.644 m
RMS EKF while radar lost: 3.634 m
RMS HG while radar lost: 2.028 m
```

On this log, the high-gain-style observer is better during radar loss than the
EKF baseline.

## Practical interpretation for the demo

This is a target vehicle estimator. It estimates the cible vehicle position
and motion from radar only.

When radar is available, the estimator corrects position and radial velocity.
When radar is lost, it predicts the target trajectory using the last estimated
velocity and acceleration. This is why the vehicle can still be tracked for a
short time without radar detections.

For a real demo, the key points to show are:

```text
1. Radar detections appear and disappear.
2. The raw radar track has holes.
3. The observer estimate remains continuous.
4. Error increases during long losses, but does not jump immediately.
5. When radar returns, the observer converges back to the target.
```
