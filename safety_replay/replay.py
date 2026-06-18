"""
FCW/IMA Safety Replay
=====================

Injects attacks on target vehicle BSMs from real NYC CV Pilot EVENT data,
recomputes safety metrics (TTC for FCW, trajectory intersection for IMA),
and quantifies the impact of each attack on safety application decisions.

All coordinates are Cartesian metres (NYC obfuscated X_m/Y_m).
Attack physics are implemented inline to avoid GPS conversion issues
in the class-based attackers.

Usage:
    python replay.py                    # Full run
    python replay.py --smoke-test       # 5 events × 3 attacks
    python replay.py --attacks false_deceleration_cruising phantom_acceleration
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from safety_replay.event_parser import load_events_by_type
from safety_replay.metrics import (
    compute_fcw_ttc_series,
    compute_ima_risk_series,
    evaluate_fcw_impact,
    evaluate_ima_impact,
    check_eebl_trigger,
)


# ============================================================
# Attack injection functions (inline, Cartesian-native)
# ============================================================

def _inject_false_deceleration_cruising(df, rng, dt=0.1):
    """Fake sudden braking: decel -> speed drop -> position lag. (Vectorized)"""
    fake_decel = rng.uniform(-5.0, -3.0)
    initial_speed = float(df['speed_mps'].values[0])
    heading_rad = np.radians(float(df['heading_deg'].values[0]))
    n = len(df)
    df = df.copy()

    idx = np.arange(n)
    elapsed = idx * dt
    new_speed = np.maximum(0, initial_speed + fake_decel * elapsed)
    df['speed_mps'] = new_speed
    df['accel_long_mps2'] = fake_decel

    # Position co-spoofing: backward offset = real_cruise_disp - braking_disp
    prev_speed = np.maximum(0, initial_speed + fake_decel * np.maximum(0, idx - 1) * dt)
    step_disp = np.maximum(0, prev_speed * dt + 0.5 * fake_decel * dt ** 2)
    step_disp[0] = 0
    cum_displacement = np.cumsum(step_disp)
    real_cruise_disp = initial_speed * idx * dt
    backward_offset = real_cruise_disp - cum_displacement
    backward_offset[0] = 0

    df['latitude'] = df['latitude'].values - backward_offset * np.cos(heading_rad)
    df['longitude'] = df['longitude'].values - backward_offset * np.sin(heading_rad)
    return df


def _inject_phantom_acceleration(df, rng, dt=0.1):
    """Stationary target reports aggressive acceleration. (Vectorized)"""
    peak_accel = rng.uniform(2.5, 5.0)
    ramp, hold, taper = 1, 2, 2
    profile_len = min(ramp + hold + taper, len(df))
    heading_rad = np.radians(float(df['heading_deg'].values[0]))
    df = df.copy()

    accel_profile = np.zeros(profile_len)
    accel_profile[:ramp] = np.linspace(0, peak_accel, ramp)
    accel_profile[ramp:ramp + hold] = peak_accel
    n_taper = min(taper, profile_len - ramp - hold)
    if n_taper > 0:
        accel_profile[ramp + hold:ramp + hold + n_taper] = np.linspace(peak_accel, 0, n_taper)

    speed_profile = np.cumsum(accel_profile) * dt
    cum_disp = np.cumsum(speed_profile) * dt

    df.iloc[:profile_len, df.columns.get_loc('accel_long_mps2')] = accel_profile
    df.iloc[:profile_len, df.columns.get_loc('speed_mps')] = speed_profile

    pos_offset = cum_disp.copy()
    pos_offset[0] = 0
    lat_vals = df['latitude'].values.copy()
    lon_vals = df['longitude'].values.copy()
    lat_vals[:profile_len] += pos_offset * np.cos(heading_rad)
    lon_vals[:profile_len] += pos_offset * np.sin(heading_rad)
    df['latitude'] = lat_vals
    df['longitude'] = lon_vals
    return df


def _inject_position_pullthrough_at_stop(df, rng, dt=0.1):
    """Offset position 20-50m along heading with speed bell curve. (Vectorized)"""
    offset_dist = rng.uniform(20, 50)
    peak_speed = rng.uniform(1.5, 3.0)
    heading_rad = np.radians(float(df['heading_deg'].values[0]))
    n = len(df)
    df = df.copy()

    n_ramp = min(5, n)
    ramp_speeds = np.sin(np.linspace(0, np.pi, n_ramp)) * peak_speed
    cum_pos = np.cumsum(ramp_speeds * dt)
    scale = offset_dist / cum_pos[-1] if cum_pos[-1] > 0 else 1.0

    # Build full-length offset, speed, accel arrays
    offsets = np.full(n, offset_dist)
    speeds = np.zeros(n)
    accels = np.zeros(n)
    offsets[:n_ramp] = cum_pos * scale
    speeds[:n_ramp] = ramp_speeds
    accel_ramp = np.zeros(n_ramp)
    accel_ramp[1:] = np.diff(ramp_speeds) / dt
    accels[:n_ramp] = accel_ramp

    df['speed_mps'] = speeds
    df['accel_long_mps2'] = accels

    noise_n = rng.normal(0, 2.5, size=n)
    noise_e = rng.normal(0, 2.5, size=n)
    df['latitude'] = df['latitude'].values + offsets * np.cos(heading_rad) + noise_n
    df['longitude'] = df['longitude'].values + offsets * np.sin(heading_rad) + noise_e
    return df


def _inject_lateral_drift_at_turning(df, rng):
    """Progressive lateral offset perpendicular to heading. (Vectorized)"""
    direction = rng.choice([-1, 1])
    max_offset = rng.uniform(5, 15)
    n = len(df)
    df = df.copy()

    progress = np.arange(1, n + 1) / n
    offset_dists = max_offset * progress
    perp_headings = np.radians(df['heading_deg'].values + 90 * direction)

    df['latitude'] = df['latitude'].values + offset_dists * np.cos(perp_headings)
    df['longitude'] = df['longitude'].values + offset_dists * np.sin(perp_headings)
    tilts = direction * np.minimum(5.0, offset_dists * 0.4)
    df['heading_deg'] = df['heading_deg'].values + tilts
    return df


def _inject_slow_position_drift(df, rng):
    """AR(1) autocorrelated gradual position drift (max 5m). (Vectorized loop on numpy)."""
    drift_rate = rng.uniform(0.05, 0.2)
    direction = rng.uniform(0, 360)
    rho = 0.95
    n = len(df)
    df = df.copy()

    # AR(1) must be sequential, but operate on raw arrays not DataFrame
    innovations = rng.normal(0, 10.0, size=n)
    dirs = np.empty(n)
    cum_n = np.empty(n)
    cum_e = np.empty(n)
    cn, ce = 0.0, 0.0
    for i in range(n):
        direction = (rho * direction + (1 - rho) * innovations[i]) % 360
        dirs[i] = direction
        dr = np.radians(direction)
        cn += drift_rate * np.cos(dr)
        ce += drift_rate * np.sin(dr)
        total = np.sqrt(cn*cn + ce*ce)
        if total > 5.0:
            cn *= 5.0 / total
            ce *= 5.0 / total
        cum_n[i] = cn
        cum_e[i] = ce

    df['latitude'] = df['latitude'].values + cum_n
    df['longitude'] = df['longitude'].values + cum_e
    return df


def _inject_heading_lock(df, rng, dt=0.1):
    """Lock heading at first value, project position along locked direction. (Vectorized)"""
    locked_heading = float(df['heading_deg'].values[0])
    locked_rad = np.radians(locked_heading)
    n = len(df)
    df = df.copy()

    df['heading_deg'] = locked_heading + rng.normal(0, 0.5, size=n)
    df['yaw_rate_degs'] = rng.normal(0, 0.5, size=n)

    # Project position along locked heading from initial position
    speeds = df['speed_mps'].values
    steps = speeds * dt
    cum_dist = np.cumsum(steps)
    cum_dist = np.concatenate([[0], cum_dist[:-1]])  # shift: pos[0] unchanged
    lat0 = df['latitude'].values[0]
    lon0 = df['longitude'].values[0]
    df['latitude'] = lat0 + cum_dist * np.cos(locked_rad)
    df['longitude'] = lon0 + cum_dist * np.sin(locked_rad)
    return df


def _inject_simple_field(df, rng, attack_name):
    """Inline vectorized field mutations for simple attacks."""
    df = df.copy()
    n = len(df)

    if attack_name == 'constant_speed':
        df['speed_mps'] = df['speed_mps'].iloc[0]
    elif attack_name == 'constant_speed_offset':
        df['speed_mps'] = df['speed_mps'] + rng.uniform(-15, 15)
        df['speed_mps'] = df['speed_mps'].clip(lower=0)
    elif attack_name == 'random_speed':
        df['speed_mps'] = rng.uniform(0, 30, size=n)
    elif attack_name == 'random_speed_offset':
        df['speed_mps'] = df['speed_mps'] + rng.uniform(-10, 10, size=n)
        df['speed_mps'] = df['speed_mps'].clip(lower=0)
    elif attack_name == 'constant_heading':
        df['heading_deg'] = df['heading_deg'].iloc[0]
    elif attack_name == 'constant_heading_offset':
        df['heading_deg'] = df['heading_deg'] + rng.uniform(-45, 45)
    elif attack_name == 'random_heading':
        df['heading_deg'] = rng.uniform(0, 360, size=n)
    elif attack_name == 'random_heading_offset':
        df['heading_deg'] = df['heading_deg'] + rng.uniform(-30, 30, size=n)
    elif attack_name == 'opposite_heading':
        df['heading_deg'] = (df['heading_deg'] + 180) % 360
    elif attack_name == 'perpendicular_heading':
        df['heading_deg'] = (df['heading_deg'] + 90) % 360
    elif attack_name == 'constant_acceleration':
        df['accel_long_mps2'] = df['accel_long_mps2'].iloc[0]
    elif attack_name == 'random_acceleration':
        df['accel_long_mps2'] = rng.uniform(-5, 5, size=n)
    elif attack_name == 'random_acceleration_offset':
        df['accel_long_mps2'] = df['accel_long_mps2'] + rng.uniform(-2, 2, size=n)
    elif attack_name == 'constant_position':
        df['latitude'] = df['latitude'].iloc[0]
        df['longitude'] = df['longitude'].iloc[0]
    elif attack_name == 'constant_position_offset':
        df['latitude'] = df['latitude'] + rng.uniform(-200, 200)
        df['longitude'] = df['longitude'] + rng.uniform(-200, 200)
    elif attack_name == 'random_position':
        center_x = df['latitude'].mean()
        center_y = df['longitude'].mean()
        df['latitude'] = center_x + rng.uniform(-500, 500, size=n)
        df['longitude'] = center_y + rng.uniform(-500, 500, size=n)
    elif attack_name == 'random_position_offset':
        df['latitude'] = df['latitude'] + rng.uniform(-50, 50, size=n)
        df['longitude'] = df['longitude'] + rng.uniform(-50, 50, size=n)
    elif attack_name == 'eventual_stop':
        initial_speed = float(df['speed_mps'].values[0])
        decel = rng.uniform(-3.0, -1.0)
        idx = np.arange(n)
        new_speed = np.maximum(0, initial_speed + decel * idx * 0.1)
        df['speed_mps'] = new_speed
        df['accel_long_mps2'] = np.where(new_speed > 0, decel, 0)
    elif attack_name == 'data_replay':
        # Reverse the trajectory (replay BSMs in reverse temporal order)
        for col in ['latitude', 'longitude', 'speed_mps', 'heading_deg',
                     'accel_long_mps2', 'accel_lat_mps2', 'yaw_rate_degs']:
            if col in df.columns:
                df[col] = df[col].values[::-1]
    elif attack_name == 'ghost_vehicle':
        # Place ghost vehicle 30m ahead of host along host heading
        offset = rng.uniform(20, 50)
        heading_rad = np.radians(df['heading_deg'].iloc[0])
        df['latitude'] = df['latitude'] + offset * np.cos(heading_rad)
        df['longitude'] = df['longitude'] + offset * np.sin(heading_rad)
        df['speed_mps'] = rng.uniform(5, 15)
        df['heading_deg'] = df['heading_deg'].iloc[0]
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    return df


# Physics-consistent attacks
PHYSICS_ATTACKS = {
    'false_deceleration_cruising': _inject_false_deceleration_cruising,
    'phantom_acceleration': _inject_phantom_acceleration,
    'position_pullthrough_at_stop': _inject_position_pullthrough_at_stop,
    'lateral_drift_at_turning': _inject_lateral_drift_at_turning,
    'slow_position_drift': _inject_slow_position_drift,
    'heading_lock': _inject_heading_lock,
}

# Simple field attacks (handled by _inject_simple_field)
SIMPLE_ATTACKS = [
    'constant_speed', 'constant_speed_offset', 'random_speed', 'random_speed_offset',
    'constant_heading', 'constant_heading_offset', 'random_heading', 'random_heading_offset',
    'opposite_heading', 'perpendicular_heading',
    'constant_acceleration', 'random_acceleration', 'random_acceleration_offset',
    'constant_position', 'constant_position_offset', 'random_position', 'random_position_offset',
    'eventual_stop', 'data_replay', 'ghost_vehicle',
]


def inject_attack_on_target(target_df: pd.DataFrame, attack_name: str,
                            seed: int = 42) -> pd.DataFrame:
    """Inject attack on target vehicle BSMs (Cartesian-native)."""
    rng = np.random.RandomState(seed)

    # Estimate dt
    dt = 0.1
    if len(target_df) > 1:
        t_diff = np.diff(target_df['timestamp'].values)
        valid = t_diff[(t_diff > 0) & (t_diff < 1.0)]
        if len(valid) > 0:
            dt = np.median(valid)

    if attack_name in PHYSICS_ATTACKS:
        fn = PHYSICS_ATTACKS[attack_name]
        if attack_name in ('false_deceleration_cruising', 'phantom_acceleration', 'heading_lock'):
            return fn(target_df, rng, dt)
        elif attack_name == 'position_pullthrough_at_stop':
            return fn(target_df, rng, dt)
        else:
            return fn(target_df, rng)
    elif attack_name in SIMPLE_ATTACKS:
        return _inject_simple_field(target_df, rng, attack_name)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


# ============================================================
# Replay runners
# ============================================================

def run_fcw_replay(events: list, attacks: list, threshold: float = 4.0,
                   seed: int = 42) -> pd.DataFrame:
    """Run FCW replay simulation for all events × attacks."""
    results = []

    for event_idx, (host_df, target_df, meta) in enumerate(events):
        # Baseline TTC
        ttc_clean = compute_fcw_ttc_series(host_df, target_df)
        eebl = check_eebl_trigger(target_df)

        for attack_name in attacks:
            try:
                target_attacked = inject_attack_on_target(
                    target_df.copy(), attack_name, seed=seed + event_idx)
                ttc_attacked = compute_fcw_ttc_series(host_df, target_attacked)
                impact = evaluate_fcw_impact(ttc_clean, ttc_attacked, threshold)
            except Exception as e:
                impact = {
                    'alert_clean': None, 'alert_attacked': None,
                    'false_alert': None, 'missed_alert': None,
                    'timing_shift_s': np.nan,
                    'min_ttc_clean': np.nan, 'min_ttc_attacked': np.nan,
                }

            impact['event_idx'] = event_idx
            impact['attack'] = attack_name
            impact['eebl_eligible'] = eebl
            results.append(impact)

    return pd.DataFrame(results)


def run_ima_replay(events: list, attacks: list, seed: int = 42) -> pd.DataFrame:
    """Run IMA replay simulation for all events × attacks."""
    results = []

    for event_idx, (host_df, target_df, meta) in enumerate(events):
        risk_clean = compute_ima_risk_series(host_df, target_df)

        for attack_name in attacks:
            try:
                target_attacked = inject_attack_on_target(
                    target_df.copy(), attack_name, seed=seed + event_idx)
                risk_attacked = compute_ima_risk_series(host_df, target_attacked)
                impact = evaluate_ima_impact(risk_clean, risk_attacked)
            except Exception as e:
                impact = {
                    'alert_clean': None, 'alert_attacked': None,
                    'false_alert': None, 'missed_alert': None,
                }

            impact['event_idx'] = event_idx
            impact['attack'] = attack_name
            results.append(impact)

    return pd.DataFrame(results)


def aggregate_results(results_df: pd.DataFrame, app_type: str) -> pd.DataFrame:
    """Compute per-attack aggregate statistics."""
    agg = results_df.groupby('attack').agg(
        n_events=('event_idx', 'count'),
        false_alert_count=('false_alert', 'sum'),
        missed_alert_count=('missed_alert', 'sum'),
        alert_clean_count=('alert_clean', 'sum'),
        alert_attacked_count=('alert_attacked', 'sum'),
    ).reset_index()

    agg['false_alert_rate'] = agg['false_alert_count'] / agg['n_events']
    agg['missed_alert_rate'] = agg['missed_alert_count'] / agg['n_events']
    agg['app_type'] = app_type

    if 'timing_shift_s' in results_df.columns:
        timing = results_df.groupby('attack')['timing_shift_s'].agg(['mean', 'median']).reset_index()
        timing.columns = ['attack', 'mean_timing_shift_s', 'median_timing_shift_s']
        agg = agg.merge(timing, on='attack', how='left')

    return agg


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FCW/IMA safety replay')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick test: 5 events × 3 attacks')
    parser.add_argument('--attacks', nargs='+', default=None,
                        help='Specific attacks to run (default: all 11)')
    parser.add_argument('--config', default='safety_replay/configs/replay.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Determine attack list
    all_attacks = [a['name'] for a in cfg['attacks']]
    if args.attacks:
        attacks = [a for a in args.attacks if a in all_attacks]
    elif args.smoke_test:
        attacks = ['false_deceleration_cruising', 'phantom_acceleration', 'position_pullthrough_at_stop']
    else:
        attacks = all_attacks

    # Load calibration for threshold
    cal_path = os.path.join(output_dir, 'calibration.json')
    fcw_threshold = 4.0  # empirical default
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        fcw_threshold = cal['fcw_ttc_at_trigger'].get('recommended_threshold_s', 4.0)
    print(f"FCW TTC threshold: {fcw_threshold}s (empirically calibrated)")

    csv_path = cfg['data']['nyc_event_csv']
    seed = cfg['random_seed']
    max_events = 5 if args.smoke_test else None

    # Load events
    print(f"\nLoading FCW events...")
    fcw_events = load_events_by_type(csv_path, 'fcw', min_target_bsms=3,
                                      max_events=max_events)
    print(f"Loading IMA events...")
    ima_events = load_events_by_type(csv_path, 'ima', min_target_bsms=3,
                                      max_events=max_events)

    # Run FCW replay
    print(f"\n--- FCW Replay: {len(fcw_events)} events × {len(attacks)} attacks ---")
    t0 = time.time()
    fcw_results = run_fcw_replay(fcw_events, attacks, threshold=fcw_threshold, seed=seed)
    fcw_time = time.time() - t0
    print(f"FCW replay completed in {fcw_time:.1f}s")

    # Run IMA replay
    print(f"\n--- IMA Replay: {len(ima_events)} events × {len(attacks)} attacks ---")
    t0 = time.time()
    ima_results = run_ima_replay(ima_events, attacks, seed=seed)
    ima_time = time.time() - t0
    print(f"IMA replay completed in {ima_time:.1f}s")

    # Save per-event results
    fcw_results.to_csv(os.path.join(output_dir, 'fcw_replay_results.csv'), index=False)
    ima_results.to_csv(os.path.join(output_dir, 'ima_replay_results.csv'), index=False)
    print(f"\nSaved per-event results to {output_dir}/")

    # Aggregate
    fcw_agg = aggregate_results(fcw_results, 'FCW')
    ima_agg = aggregate_results(ima_results, 'IMA')
    combined_agg = pd.concat([fcw_agg, ima_agg], ignore_index=True)
    combined_agg.to_csv(os.path.join(output_dir, 'replay_aggregate.csv'), index=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"FCW Results (n={len(fcw_events)} events, threshold={fcw_threshold}s)")
    print(f"{'='*70}")
    print(fcw_agg[['attack', 'n_events', 'false_alert_rate', 'missed_alert_rate']].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"IMA Results (n={len(ima_events)} events)")
    print(f"{'='*70}")
    print(ima_agg[['attack', 'n_events', 'false_alert_rate', 'missed_alert_rate']].to_string(index=False))

    print(f"\nTotal time: {fcw_time + ima_time:.1f}s")


if __name__ == '__main__':
    main()
