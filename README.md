# High Gain Observer vs Kalman Filter (MATLAB)

This project compares nonlinear High-Gain Observer variants against an Extended Kalman Filter (EKF) for vehicle-like kinematic state estimation.

It includes:
- Synthetic simulation scripts (no real-data dependency)
- Real-data comparison scripts (using Data.mat)
- Core observer/model functions and EKF class

## Recommended Main Scripts

Use one of these as the main entry, depending on your goal:

1. `Compare_method_to_kalman.m` (recommended primary entry)
- Best overall demo for this repository.
- Uses real data from `Data.mat`.
- Compares Multi-Output High-Gain Observer vs EKF.

2. `comparaison_SHGO.m`
- Real-data comparison between Multi-Output High-Gain Observer and standard SHGO-style high-gain observer.

3. `main.m`
- Standalone synthetic simulation for the base High-Gain Observer pipeline.
- Good starting point to understand the method before real-data scripts.

## Full Script Overview

### Entry Scripts (runnable)
- `main.m`: Synthetic kinematic simulation + high-gain observer.
- `Compare_method_to_kalman.m`: Real-data benchmark (Multi-Output HG vs EKF).
- `comparaison_SHGO.m`: Real-data benchmark (Multi-Output HG vs SHGO).
- `First_method.m`: Synthetic/legacy method-1 observer experiment, includes comparison with `Meth2.mat`.
- `second_method.m`: Legacy method-2 script for real data.
- `LMI_test.m`: Separate LMI observer design example on a link-robot model.

### Core Functions and Class
- `HighGainObserver.m`: Base high-gain observer dynamics.
- `HighGainObserver_method1.m`: Method-1 observer variant with extra speed-magnitude correction.
- `HighGainObserver_method2.m`: Method-2 multi-output observer variant (position + velocity correction terms).
- `ExtendedKalmanFilter.m`: EKF class with `propagate` and `update` methods.
- `CinematicModel.m`: Nonlinear vehicle kinematic model.
- `CinematicModelObs.m`: Observer form of kinematic model with correction term.
- `CinematicModelTransformed.m`: State-transformed dynamics.
- `Phi.m`: Nonlinear term used in transformed observer model.
- `Proj.m`: Component-wise projection/saturation helper.
- `NormlizeAngle.m`: Angle wrap helper to keep yaw in [-pi, pi].
- `stateModel.m`, `measureModel.m`: Additional generic state/measurement model helpers.
- `LinkRobot.m`: Dynamics used by `LMI_test.m`.
- `product.m`: Cartesian-product utility.

## Mathematical Form (Easy View)

### 1) Vehicle Kinematic Model (used in `CinematicModel.m`)

State:
$$
x = \begin{bmatrix}X & Y & \psi & \delta\end{bmatrix}^T
$$

Dynamics:
$$
\dot X = v\cos(\psi), \quad
\dot Y = v\sin(\psi), \quad
\dot \psi = \frac{v}{L_f}\tan(\delta), \quad
\dot \delta = 0
$$

Measured output (GPS-like):
$$
y = \begin{bmatrix}X & Y\end{bmatrix}^T + \eta
$$

### 2) Transformed Nonlinear Model (used by HG observer design)

Transformed state:
$$
z = \begin{bmatrix}z_1 & z_2 & z_3 & z_4 & z_5 & z_6\end{bmatrix}^T
$$

With
$$
\alpha(z) = \frac{-z_5 z_3 + z_2 z_6}{v^2}
$$

Dynamics (`CinematicModelTransformed.m`):
$$
\dot z_1 = z_2,\; \dot z_2 = z_3,\; \dot z_3 = -\alpha z_6,
$$
$$
\dot z_4 = z_5,\; \dot z_5 = z_6,\; \dot z_6 = \alpha z_3
$$

Equivalent compact form used in code:
$$
\dot z = A z + B\,\Phi(z,v)
$$
where
$$
\Phi(z,v)=
\begin{bmatrix}
-\frac{1}{v^2}(-z_5z_3+z_2z_6)z_6 \\
\frac{1}{v^2}(-z_5z_3+z_2z_6)z_3
\end{bmatrix}
$$

### 3) Base High-Gain Observer (`HighGainObserver.m`)

Let $\hat z$ be the estimated transformed state, and $y_c$ the interpolated measurement at current time.

When measurement is available:
$$
\dot{\hat z} = A\hat z + B\,\Phi(\hat z,v) + T K (y_c - C\hat z)
$$

When measurement is missing (NaN):
$$
\dot{\hat z} = A\hat z + B\,\Phi(\hat z,v)
$$

Optional projection/saturation in code:
$$
\dot{\hat z} \leftarrow \mathrm{Proj}(\dot{\hat z})
$$

### 4) Method-1 Observer (`HighGainObserver_method1.m`)

$$
\dot{\hat z} = A\hat z + B\,\Phi(\hat z,v) + T K (y_c - C\hat z) + M\left(v-\sqrt{\hat z_2^2+\hat z_5^2}\right)
$$

### 5) Method-2 Multi-Output Observer (`HighGainObserver_method2.m`)

Velocity correction term:
$$
\Delta_v=
\begin{bmatrix}
v_x-\hat z_2 \\
v_y-\hat z_5
\end{bmatrix}
$$

When position measurement is available:
$$
\dot{\hat z} = A\hat z + B\,\Phi(\hat z,v) + T K (y_c - C\hat z) + T M\Delta_v
$$

When position measurement is missing:
$$
\dot{\hat z} = A\hat z + B\,\Phi(\hat z,v) + T M\Delta_v
$$

### 6) EKF Model Used in Comparison (`ExtendedKalmanFilter.m`)

State:
$$
\mu = \begin{bmatrix}x & y & \theta\end{bmatrix}^T, \quad u=\begin{bmatrix}v & \omega\end{bmatrix}^T
$$

Propagation with $r=\frac{v}{\omega}$:
$$
x_{k+1}=x_k-r\sin\theta_k+r\sin(\theta_k+\omega\Delta t)
$$
$$
y_{k+1}=y_k+r\cos\theta_k-r\cos(\theta_k+\omega\Delta t)
$$
$$
	heta_{k+1}=\theta_k+\omega\Delta t
$$

Measurement model:
$$
z_k = \begin{bmatrix}x_k & y_k\end{bmatrix}^T + \nu_k
$$


## Function Dependencies by Main Script

### `main.m`
Calls:
- `CinematicModel`
- `NormlizeAngle`
- `HighGainObserver`

`HighGainObserver` internally calls:
- `Phi`
- `Proj` (only when optional saturation vertices are provided)

### `Compare_method_to_kalman.m`
Calls:
- `HighGainObserver_method2`
- `ExtendedKalmanFilter`
- `NormlizeAngle`

`HighGainObserver_method2` internally calls:
- `Phi`
- `Proj` (when optional vertices are provided)

### `comparaison_SHGO.m`
Calls:
- `HighGainObserver_method2`
- `HighGainObserver`

Internal downstream calls:
- `Phi`
- `Proj` (when optional vertices are provided)

### `First_method.m`
Calls:
- `CinematicModel`
- `NormlizeAngle`
- `HighGainObserver_method1`
- Loads `Meth2.mat` for side-by-side comparison

`HighGainObserver_method1` internally calls:
- `Phi`
- `Proj` (when optional vertices are provided)

### `LMI_test.m`
Calls:
- `LinkRobot`

## Data Files

- `Data.mat`: Required by `Compare_method_to_kalman.m`, `comparaison_SHGO.m`, and `second_method.m`.
- `Meth2.mat`: Used by `First_method.m` for method comparison plots.
- `IntersectionStraight.mat`: Present in repository, not directly used in the main scripts above.

## Requirements

MATLAB toolboxes and external packages used by scripts:
- Symbolic Math Toolbox (`syms`, `jacobian`)
- Statistics and Machine Learning Toolbox (`normrnd`)
- ODE solvers (`ode45`, base MATLAB)
- YALMIP (`sdpvar`, `optimize`) for LMI setup
- An SDP solver configured for YALMIP (for example SDPT3, SeDuMi, or MOSEK)

## Quick Start

1. Open MATLAB.
2. Set current folder to this project directory.
3. Run one of:
- `Compare_method_to_kalman`
- `comparaison_SHGO`
- `main`

## Notes

- `Compare_method_to_kalman.m` is the best candidate for the project "main" when your objective is real-data method comparison.
- `main.m` is the best educational entry for synthetic simulation and understanding model/observer flow.
- `second_method.m` appears to be a legacy script and may require argument updates to match the current `HighGainObserver_method2` function signature.
