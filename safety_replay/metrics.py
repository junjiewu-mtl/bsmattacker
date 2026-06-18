"""
Safety Metrics: FCW TTC & IMA Collision Prediction
====================================================

Computes Forward Collision Warning (TTC) and Intersection Movement Assist
(trajectory intersection) metrics from host/target BSM DataFrames.

Coordinates are Cartesian metres (NYC CV Pilot obfuscated X_m/Y_m).

References:
    DOT HS 811 492B — VSC-A Final Report (NHTSA, 2011)
    SAE J2945/1 — BSM requirements and recommended thresholds
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def _interpolate_target(host_df: pd.DataFrame, target_df: pd.DataFrame,
                        columns: list) -> pd.DataFrame:
    """Interpolate target vehicle state at host BSM timestamps.

    Uses linear interpolation on T_s (timestamp). Only interpolates within
    the overlapping time range — extrapolation is NaN.

    Returns DataFrame indexed like host_df with interpolated target columns.
    """
    host_t = host_df['timestamp'].values
    target_t = target_df['timestamp'].values

    # Only interpolate within target time range
    t_min, t_max = target_t.min(), target_t.max()

    result = pd.DataFrame(index=host_df.index)
    result['timestamp'] = host_t

    for col in columns:
        if col in target_df.columns:
            interp_vals = np.interp(host_t, target_t, target_df[col].values,
                                    left=np.nan, right=np.nan)
            # Mask outside target range
            interp_vals[host_t < t_min] = np.nan
            interp_vals[host_t > t_max] = np.nan
            result[f'target_{col}'] = interp_vals
        else:
            result[f'target_{col}'] = np.nan

    return result


def compute_fcw_ttc_series(host_df: pd.DataFrame,
                           target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute TTC at each host BSM timestamp.

    TTC = distance / closing_speed, where closing_speed is the component
    of relative velocity along the host-to-target bearing axis.

    Cartesian: distance = sqrt(dx² + dy²), no GPS conversion needed.

    Returns DataFrame with columns:
        T_s, distance_m, closing_speed_mps, ttc_s
    """
    # Interpolate target state at host timestamps
    interp_cols = ['latitude', 'longitude', 'speed_mps', 'heading_deg']
    interp = _interpolate_target(host_df, target_df, interp_cols)

    host_x = host_df['latitude'].values   # Cartesian X_m
    host_y = host_df['longitude'].values  # Cartesian Y_m
    host_speed = host_df['speed_mps'].values
    host_heading = np.radians(host_df['heading_deg'].values)

    target_x = interp['target_latitude'].values
    target_y = interp['target_longitude'].values
    target_speed = interp['target_speed_mps'].values
    target_heading = np.radians(interp['target_heading_deg'].values)

    # Distance (Cartesian Euclidean)
    dx = target_x - host_x
    dy = target_y - host_y
    distance = np.sqrt(dx**2 + dy**2)

    # Bearing from host to target
    bearing = np.arctan2(dx, dy)  # standard math bearing

    # Project speeds onto host-to-target axis
    host_approach = host_speed * np.cos(host_heading - bearing)
    target_recede = target_speed * np.cos(target_heading - bearing)
    closing_speed = host_approach - target_recede

    # TTC = distance / closing_speed (only meaningful when closing)
    ttc = np.full_like(distance, np.inf)
    valid = closing_speed > 0.1  # 0.1 m/s threshold to avoid div-by-near-zero
    ttc[valid] = distance[valid] / closing_speed[valid]

    result = pd.DataFrame({
        'T_s': host_df['timestamp'].values,
        'distance_m': distance,
        'closing_speed_mps': closing_speed,
        'ttc_s': ttc,
    })

    return result


def compute_ima_risk_series(host_df: pd.DataFrame,
                            target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute IMA intersection collision risk at each host timestamp.

    Model: project each vehicle's trajectory as a ray from current position
    along current heading. Find the closest approach point and time each
    vehicle needs to reach it. Alert when:
      1. min_approach_distance < lane_width (3.5m)
      2. Both vehicles reach the approach point within overlap window

    Returns DataFrame with columns:
        T_s, min_approach_dist_m, host_tti_s, target_tti_s, alert_active
    """
    interp_cols = ['latitude', 'longitude', 'speed_mps', 'heading_deg']
    interp = _interpolate_target(host_df, target_df, interp_cols)

    host_x = host_df['latitude'].values
    host_y = host_df['longitude'].values
    host_speed = host_df['speed_mps'].values
    host_heading = np.radians(host_df['heading_deg'].values)

    target_x = interp['target_latitude'].values
    target_y = interp['target_longitude'].values
    target_speed = interp['target_speed_mps'].values
    target_heading = np.radians(interp['target_heading_deg'].values)

    n = len(host_df)
    min_dist = np.full(n, np.inf)
    host_tti = np.full(n, np.inf)
    target_tti = np.full(n, np.inf)

    for i in range(n):
        if np.isnan(target_x[i]):
            continue

        # Direction unit vectors
        h_dx = np.sin(host_heading[i])
        h_dy = np.cos(host_heading[i])
        t_dx = np.sin(target_heading[i])
        t_dy = np.cos(target_heading[i])

        # Position difference
        wx = host_x[i] - target_x[i]
        wy = host_y[i] - target_y[i]

        # Parametric closest approach: solve for t_h, t_t
        # host_pos(t_h) = (host_x + h_dx*v_h*t_h, host_y + h_dy*v_h*t_h)
        # target_pos(t_t) = (target_x + t_dx*v_t*t_t, target_y + t_dy*v_t*t_t)
        # Minimize || host_pos(t_h) - target_pos(t_t) ||²

        v_h = max(host_speed[i], 0.1)  # avoid zero speed
        v_t = max(target_speed[i], 0.1)

        # Direction vectors scaled by speed
        a = h_dx * v_h
        b = h_dy * v_h
        c = t_dx * v_t
        d = t_dy * v_t

        # Solve 2x2 system for minimum distance
        # d/dt_h: a*(wx + a*t_h - c*t_t) + b*(wy + b*t_h - d*t_t) = 0
        # d/dt_t: -c*(wx + a*t_h - c*t_t) - d*(wy + b*t_h - d*t_t) = 0
        A11 = a*a + b*b
        A12 = -(a*c + b*d)
        A21 = -(a*c + b*d)
        A22 = c*c + d*d
        b1 = -(a*wx + b*wy)
        b2 = c*wx + d*wy

        det = A11*A22 - A12*A21
        if abs(det) < 1e-10:
            # Parallel trajectories — use perpendicular distance
            cross = abs(h_dx * (target_y[i] - host_y[i]) - h_dy * (target_x[i] - host_x[i]))
            min_dist[i] = cross
            continue

        t_h = (b1*A22 - b2*A12) / det
        t_t = (A11*b2 - A21*b1) / det

        if t_h < 0 or t_t < 0:
            # Vehicles diverging — no future intersection
            continue

        # Closest approach distance
        px = wx + a*t_h - c*t_t
        py = wy + b*t_h - d*t_t
        min_dist[i] = np.sqrt(px*px + py*py)
        host_tti[i] = t_h
        target_tti[i] = t_t

    # Alert when distance < 3.5m and both arrive within threshold
    alert = (min_dist < 3.5) & (host_tti < 30) & (target_tti < 30)

    return pd.DataFrame({
        'T_s': host_df['timestamp'].values,
        'min_approach_dist_m': min_dist,
        'host_tti_s': host_tti,
        'target_tti_s': target_tti,
        'alert_active': alert,
    })


def evaluate_fcw_impact(ttc_clean: pd.DataFrame, ttc_attacked: pd.DataFrame,
                        threshold: float = 2.6) -> dict:
    """Compare clean vs attacked TTC series and classify impact.

    Returns dict with:
        false_alert: attack causes alert when clean had no alert
        missed_alert: attack suppresses alert when clean had alert
        timing_shift_s: change in alert onset time (negative = earlier)
        min_ttc_clean, min_ttc_attacked: minimum TTC values
    """
    # Clean: was there an alert?
    clean_alert_mask = ttc_clean['ttc_s'] <= threshold
    attacked_alert_mask = ttc_attacked['ttc_s'] <= threshold

    clean_had_alert = clean_alert_mask.any()
    attacked_has_alert = attacked_alert_mask.any()

    min_ttc_clean = ttc_clean['ttc_s'].replace(np.inf, np.nan).min()
    min_ttc_attacked = ttc_attacked['ttc_s'].replace(np.inf, np.nan).min()

    # Alert onset times
    clean_onset = ttc_clean.loc[clean_alert_mask, 'T_s'].min() if clean_had_alert else np.nan
    attacked_onset = ttc_attacked.loc[attacked_alert_mask, 'T_s'].min() if attacked_has_alert else np.nan

    timing_shift = np.nan
    if not np.isnan(clean_onset) and not np.isnan(attacked_onset):
        timing_shift = attacked_onset - clean_onset  # negative = earlier alert

    return {
        'alert_clean': clean_had_alert,
        'alert_attacked': attacked_has_alert,
        'false_alert': (not clean_had_alert) and attacked_has_alert,
        'missed_alert': clean_had_alert and (not attacked_has_alert),
        'timing_shift_s': timing_shift,
        'min_ttc_clean': min_ttc_clean if not np.isnan(min_ttc_clean) else np.inf,
        'min_ttc_attacked': min_ttc_attacked if not np.isnan(min_ttc_attacked) else np.inf,
    }


def evaluate_ima_impact(risk_clean: pd.DataFrame, risk_attacked: pd.DataFrame,
                        tti_threshold: float = 3.0) -> dict:
    """Compare clean vs attacked IMA risk and classify impact."""
    clean_alert = risk_clean['alert_active'].any()
    attacked_alert = risk_attacked['alert_active'].any()

    return {
        'alert_clean': clean_alert,
        'alert_attacked': attacked_alert,
        'false_alert': (not clean_alert) and attacked_alert,
        'missed_alert': clean_alert and (not attacked_alert),
    }


def check_eebl_trigger(target_df: pd.DataFrame,
                        decel_threshold: float = -3.92) -> bool:
    """Check if target vehicle has deceleration ≥ 0.4g at any point.

    Returns True if this event also meets EEBL trigger criteria.
    """
    if target_df is None or target_df.empty:
        return False
    return bool((target_df['accel_long_mps2'] <= decel_threshold).any())


if __name__ == '__main__':
    """Quick validation: compute TTC for first 10 FCW events."""
    import yaml
    from safety_replay.event_parser import load_events_by_type

    with open('safety_replay/configs/replay.yaml') as f:
        cfg = yaml.safe_load(f)

    events = load_events_by_type(cfg['data']['nyc_event_csv'], 'fcw',
                                  min_target_bsms=3, max_events=10)

    print(f"\nTTC validation on {len(events)} FCW events:")
    print(f"{'Event':<8} {'MinTTC':>8} {'TTC@0':>8} {'Alert':>6} {'EEBL':>6}")
    print('-' * 40)

    for i, (host_df, target_df, meta) in enumerate(events):
        ttc_series = compute_fcw_ttc_series(host_df, target_df)
        # TTC at T_s closest to 0
        trigger_idx = (ttc_series['T_s'].abs()).idxmin()
        ttc_at_trigger = ttc_series.loc[trigger_idx, 'ttc_s']
        min_ttc = ttc_series['ttc_s'].replace(np.inf, np.nan).min()
        alert = min_ttc <= 2.6 if not np.isnan(min_ttc) else False
        eebl = check_eebl_trigger(target_df)

        print(f"{i:<8} {min_ttc:>8.2f} {ttc_at_trigger:>8.2f} {str(alert):>6} {str(eebl):>6}")
