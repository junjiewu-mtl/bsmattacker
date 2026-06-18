#!/usr/bin/env python3
"""VeReMi Extension Sensor Noise Model (Kamel et al., IEEE ICC 2020).

Applies realistic sensor noise to BSM attack values, matching the
VeReMi Extension dataset's error model. Without this, BSMAttacker
injects mathematically clean values that are easier to detect than
real-world attacks would be.

VeReMi Extension sensor error model:
  Position: GPS ±5m correlated random walk (σ_pos = 0.03 * E₀ ≈ 5m)
  Speed:    proportional error (σ_speed = 0.00016 * V)
  Heading:  velocity-dependent U([-20,20]) * exp(-0.1 * V) degrees
  Accel:    inferred from velocity noise (Δv_noise / dt)
  Yaw rate: heading noise derivative (Δheading_noise / dt)

Reference:
  Kamel et al. (2020), "VeReMi Extension: A Dataset for Comparable
  Evaluation of Misbehavior Detection in VANETs", IEEE ICC 2020.
  Table I: Sensor Error Model Parameters.
"""

import numpy as np
import pandas as pd


def apply_sensor_noise(
    df: pd.DataFrame,
    attack_mask: pd.Series,
    seed: int = 42,
    cartesian: bool = True,
    noise_mode: str = "realistic",
) -> pd.DataFrame:
    """Add sensor noise to attacked rows.

    Args:
        df: DataFrame with BSM data (after attack injection).
        attack_mask: Boolean mask where True = attacked row.
        seed: Random seed for reproducibility.
        cartesian: True if positions are in metres (VeReMi/SUMO),
                   False if in GPS degrees (real OBU data).
        noise_mode: "realistic" = AR(1) correlated (Run 1/3/4 default),
                    "veremi" = i.i.d. Gaussian per F2MD source (Run 2 Set B).

    Returns:
        DataFrame with noise added to attacked rows.
    """
    if noise_mode == "veremi":
        return _apply_veremi_noise(df, attack_mask, seed, cartesian)

    df = df.copy()
    rng = np.random.RandomState(seed)
    n_attacked = attack_mask.sum()
    if n_attacked == 0:
        return df

    idx = attack_mask[attack_mask].index

    # --- Position noise: per-device correlated GPS random walk ---
    # VeReMi Extension: σ_pos ≈ 5m (from E₀ ≈ 167m, σ = 0.03 * E₀)
    # PERF: prior loop did `(df.loc[idx,"device_id"]==dev_id).values`
    # per device (O(n_attacked) scan × n_devices ≈ 32 B ops on symmetric-noise
    # runs). Precompute per-device contiguous index ranges once and iterate
    # groups instead. Numerical results differ from the old code because the
    # RNG draw order changes (sorted-by-device vs first-appearance); for our
    # symmetric-noise use case the noise realisation is only required to be
    # statistically equivalent, which it is.
    sigma_pos_m = 5.0  # metres
    if cartesian:
        pos_params = [("latitude", sigma_pos_m), ("longitude", sigma_pos_m)]
    else:
        pos_params = [
            ("latitude", sigma_pos_m / 111_320),
            ("longitude", sigma_pos_m / 78_000),
        ]

    # Precompute group index ranges once (shared across both position columns).
    if "device_id" in df.columns:
        device_arr = df.loc[idx, "device_id"].values
        sort_order = np.argsort(device_arr, kind="stable")
        sorted_devs = device_arr[sort_order]
        # Group boundaries in sorted-order
        if len(sorted_devs) == 0:
            group_starts = np.array([0], dtype=np.int64)
        else:
            boundary = np.nonzero(sorted_devs[1:] != sorted_devs[:-1])[0] + 1
            group_starts = np.concatenate(
                [np.array([0], dtype=np.int64), boundary,
                 np.array([len(sorted_devs)], dtype=np.int64)]
            )
    else:
        sort_order = None
        group_starts = None

    for col, sigma in pos_params:
        noise = np.zeros(n_attacked)
        if sort_order is not None:
            for gi in range(len(group_starts) - 1):
                s, e = int(group_starts[gi]), int(group_starts[gi + 1])
                noise[sort_order[s:e]] = _correlated_noise(e - s, sigma, rng)
        else:
            noise = _correlated_noise(n_attacked, sigma, rng)
        df.loc[idx, col] += noise

    # --- Speed noise: proportional to velocity ---
    # VeReMi Extension: σ_speed = 0.00016 * V (no floor — faithful to paper)
    speed_col = "speed_mps"
    if speed_col in df.columns:
        speed_vals = df.loc[idx, speed_col].values
        sigma_speed = 0.00016 * np.abs(speed_vals)
        # Only add noise where sigma > 0 (stationary gets zero noise, per VeReMi)
        nonzero = sigma_speed > 0
        noise = np.zeros(n_attacked)
        noise[nonzero] = rng.normal(0, sigma_speed[nonzero])
        df.loc[idx, speed_col] += noise
        df.loc[idx, speed_col] = df.loc[idx, speed_col].clip(lower=0)

    # --- Heading noise: velocity-dependent ---
    # VeReMi Extension: heading_noise ~ U([-20, 20]) * exp(-0.1 * V)
    # At V=0: ±20°, at V=10 m/s: ±7.4°, at V=30 m/s: ±1.0°
    heading_col = "heading_deg"
    if heading_col in df.columns:
        speed_for_heading = df.loc[idx, speed_col].values if speed_col in df.columns else np.zeros(n_attacked)
        heading_amplitude = 20.0 * np.exp(-0.1 * np.abs(speed_for_heading))
        heading_noise = rng.uniform(-1, 1, n_attacked) * heading_amplitude
        df.loc[idx, heading_col] = (df.loc[idx, heading_col].values + heading_noise) % 360

    # --- Acceleration noise: inferred from velocity noise ---
    # σ_accel ≈ σ_speed / dt, with dt = 0.1s → σ_accel ≈ 1.0 m/s²
    # Use fixed σ = 0.5 m/s² (conservative, matches real OBU noise)
    sigma_accel = 0.5
    for accel_col in ["accel_long_mps2", "accel_lat_mps2"]:
        if accel_col in df.columns:
            df.loc[idx, accel_col] += rng.normal(0, sigma_accel, n_attacked)

    # --- Yaw rate noise: heading noise derivative ---
    # σ_yaw ≈ heading_noise_amplitude / dt
    # Use fixed σ = 2.0 deg/s (matches a static-lab session measured noise)
    yaw_col = "yaw_rate_degs"
    if yaw_col not in df.columns:
        yaw_col = "yaw_rate_dps"
    if yaw_col in df.columns:
        sigma_yaw = 2.0
        df.loc[idx, yaw_col] += rng.normal(0, sigma_yaw, n_attacked)

    return df


def _apply_veremi_noise(
    df: pd.DataFrame,
    attack_mask: pd.Series,
    seed: int = 42,
    cartesian: bool = True,
) -> pd.DataFrame:
    """VeReMi-faithful i.i.d. Gaussian noise (no temporal correlation).

    Source: F2MD MDModules.cc, genGaussianNoise(0, conf/3).
    The /3 maps confidence interval to sigma via 99.7% rule (+/-3sigma = conf).

    Parameters from F2MD source:
        confPos  = 10m   -> sigma_pos  = 10/3 = 3.33m
        confSpd  = 0.05  -> sigma_spd  = 0.05/3 = 0.017 m/s
        confHea  = 20    -> sigma_hea  = 20/3 = 6.67 deg
        confAcc  = 0.01  -> sigma_acc  = 0.01/3 = 0.003 m/s2
    """
    df = df.copy()
    rng = np.random.RandomState(seed)
    n_attacked = attack_mask.sum()
    if n_attacked == 0:
        return df

    idx = attack_mask[attack_mask].index

    # Position: i.i.d. Gaussian, sigma = confPos/3 = 3.33m
    sigma_pos_m = 10.0 / 3.0
    if cartesian:
        pos_params = [("latitude", sigma_pos_m), ("longitude", sigma_pos_m)]
    else:
        pos_params = [
            ("latitude", sigma_pos_m / 111_320),
            ("longitude", sigma_pos_m / 78_000),
        ]
    for col, sigma in pos_params:
        if col in df.columns:
            df.loc[idx, col] += rng.normal(0, sigma, n_attacked)

    # Speed: i.i.d. Gaussian, sigma = confSpd/3 = 0.017 m/s
    speed_col = "speed_mps"
    if speed_col in df.columns:
        sigma_spd = 0.05 / 3.0
        df.loc[idx, speed_col] += rng.normal(0, sigma_spd, n_attacked)
        df.loc[idx, speed_col] = df.loc[idx, speed_col].clip(lower=0)

    # Heading: i.i.d. Gaussian, sigma = confHea/3 = 6.67 deg
    heading_col = "heading_deg"
    if heading_col in df.columns:
        sigma_hea = 20.0 / 3.0
        df.loc[idx, heading_col] = (
            df.loc[idx, heading_col].values + rng.normal(0, sigma_hea, n_attacked)
        ) % 360

    # Acceleration: i.i.d. Gaussian, sigma = confAcc/3 = 0.003 m/s2
    sigma_acc = 0.01 / 3.0
    for accel_col in ["accel_long_mps2", "accel_lat_mps2"]:
        if accel_col in df.columns:
            df.loc[idx, accel_col] += rng.normal(0, sigma_acc, n_attacked)

    # Yaw rate: derive from heading sigma / dt (dt=1s for BSM rate)
    yaw_col = "yaw_rate_degs"
    if yaw_col not in df.columns:
        yaw_col = "yaw_rate_dps"
    if yaw_col in df.columns:
        sigma_yaw = 20.0 / 3.0  # same as heading per message
        df.loc[idx, yaw_col] += rng.normal(0, sigma_yaw, n_attacked)

    return df


def _correlated_noise(n: int, sigma: float, rng: np.random.RandomState,
                      rho: float = 0.95) -> np.ndarray:
    """Generate AR(1) correlated noise (GPS random walk).

    Args:
        n: Number of samples.
        sigma: Standard deviation of the noise.
        rho: AR(1) autocorrelation coefficient (0.95 = strong correlation).
        rng: Random state.

    Returns:
        Array of correlated noise values.
    """
    innovation_sigma = sigma * np.sqrt(1 - rho**2)
    noise = np.zeros(n)
    noise[0] = rng.normal(0, sigma)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + rng.normal(0, innovation_sigma)
    return noise
