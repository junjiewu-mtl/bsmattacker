"""
Phantom Acceleration Attack (v1.0)
===================================

This attack targets vehicles in STATIONARY or slow-speed scenarios, falsely
reporting sudden aggressive acceleration. The vehicle appears to launch forward
when it is actually stopped or crawling, potentially causing rear vehicles to
yield or change lanes unnecessarily.

Attack Mechanism:
- Target: Messages with Scenario_Label = 'Stationary_Brief' (brief stops, < 5 s).
  Falls back to flat 'Stationary' if duration-aware labeling hasn't been run.
  Does NOT target 'Stationary_Wait' — that is liar_at_light's domain.
- Modification: Injects a realistic acceleration profile (0 → peak_accel → taper)
  with co-spoofed speed and position that match the reported acceleration
- Realism: Physics-consistent speed = ∫accel·dt, position = ∫speed·dt

Co-spoofing (multi-field consistency):
- Speed: integrated from spoofed acceleration profile
- Position: integrated from spoofed speed along original heading
- Heading: preserved from real data (straight-line launch)
- Acceleration: trapezoidal ramp-up → hold → taper profile

Literature Reference:
- Kamel et al. (2020), "Simulation Framework for Misbehavior Detection in
  Vehicular Networks", IEEE TVT — acceleration plausibility checks
- VeReMi Extension "Random Acceleration" variant, but scenario-aware and
  physics-consistent across speed/position/acceleration fields

Use Case:
This attack exploits cooperative awareness: following vehicles trust BSM accel
data for predictive collision avoidance (e.g., EEBL). A phantom launch from a
stopped vehicle creates false "vehicle entering traffic" events.

"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class PhantomAccelerationAttacker(BaseAttacker):
    """
    Implements the 'Phantom Acceleration' attack for stationary/slow vehicles.

    Injects a physics-consistent acceleration → speed → position profile
    that makes a stopped vehicle appear to launch forward aggressively.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Phantom Acceleration", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.3,
                     peak_accel_range: tuple = (2.5, 5.0),
                     ramp_msgs: int = 1,
                     hold_msgs: int = 2,
                     taper_msgs: int = 2,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Phantom Acceleration' attacks into stationary messages.

        The acceleration profile is trapezoidal:
          [ramp: 0→peak] [hold: peak] [taper: peak→0]
        Speed and position are integrated from acceleration for consistency.

        Args:
            df: Input DataFrame with BSM data (must have 'Scenario_Label' column)
            attack_ratio: Proportion of eligible vehicles to attack (0.0 to 1.0)
            peak_accel_range: Tuple of (min, max) peak acceleration in m/s²
            ramp_msgs: Number of messages for acceleration ramp-up phase
            hold_msgs: Number of messages to hold peak acceleration
            taper_msgs: Number of messages for acceleration taper-down phase
            target_vehicles: List of Device_IDs to attack (None = attack all)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        time_col = get_column_name(df, 'timestamp') or 'Tx_Timestamp'
        lat_col = get_column_name(df, 'latitude') or 'Latitude_deg'
        lon_col = get_column_name(df, 'longitude') or 'Longitude_deg'
        heading_col = get_column_name(df, 'heading') or 'Heading_deg'
        speed_col = get_column_name(df, 'speed') or 'Speed_mps'
        accel_col = get_column_name(df, 'accel_long') or 'Accel_Long_mps2'
        scenario_col = get_column_name(df, 'scenario_label') or 'Scenario_Label'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        if scenario_col not in df.columns:
            raise ValueError(f"DataFrame must have '{scenario_col}' column. Run scenario labeling first.")

        attack_mask = pd.Series(False, index=df.index)

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        # Total attack profile length
        profile_len = ramp_msgs + hold_msgs + taper_msgs

        # Target Stationary_Brief only (brief stops < 5s) to avoid overlap
        # with liar_at_light which owns Stationary_Wait. Fall back to flat
        # 'Stationary' if duration-aware labeling hasn't been applied.
        n_brief = (df[scenario_col] == 'Stationary_Brief').sum()
        n_flat = (df[scenario_col] == 'Stationary').sum()
        if n_brief > 0:
            target_label = 'Stationary_Brief'
        elif n_flat > 0:
            import warnings
            warnings.warn(
                f"phantom_acceleration: 0 Stationary_Brief rows but {n_flat} flat "
                f"'Stationary' rows found. Falling back to 'Stationary'. "
                f"Run duration-aware labeling for precise targeting.",
                stacklevel=2,
            )
            target_label = 'Stationary'
        else:
            target_label = 'Stationary_Brief'  # will match 0 rows → no injection

        # phantom-launch is credible ONLY from stop-controlled
        # intersections (signal or stop sign) — mid-traffic hesitations produce
        # distinct kinematic profiles (MDPI Sustainability 2025 17/20/9332).
        # Deployment: signal-controlled via MAP signal group; stop-controlled
        # is "MAP-absent + visible stop sign" (attacker reads OSM tag). Offline
        # proxy: is_signalized OR is_stop_controlled. VeReMi fallback:
        # kinematic-only Stationary_Brief.
        has_osm = 'is_signalized' in df.columns and 'is_stop_controlled' in df.columns
        if has_osm:
            context_ok = (
                df['is_signalized'].fillna(False).astype(bool)
                | df['is_stop_controlled'].fillna(False).astype(bool)
            )
        elif 'in_intersection_zone' in df.columns:
            context_ok = df['in_intersection_zone'].fillna(False).astype(bool)
        else:
            context_ok = pd.Series(True, index=df.index)

        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (
                df[scenario_col] == target_label
            ) & context_ok
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) < profile_len:
                continue

            segments = self._find_contiguous_segments(vehicle_indices)
            valid_segments = [s for s in segments if len(s) >= profile_len]

            if not valid_segments:
                continue

            # Attack a fraction of valid segments
            n_segments_to_attack = max(1, int(len(valid_segments) * attack_ratio))
            segments_to_attack = valid_segments[:n_segments_to_attack]

            for segment in segments_to_attack:
                attack_indices = segment[:profile_len]

                peak_accel = np.random.uniform(peak_accel_range[0], peak_accel_range[1])

                # Build trapezoidal acceleration profile
                accel_profile = np.zeros(profile_len)
                # Ramp: linear 0 → peak
                accel_profile[:ramp_msgs] = np.linspace(0, peak_accel, ramp_msgs)
                # Hold: constant peak
                accel_profile[ramp_msgs:ramp_msgs + hold_msgs] = peak_accel
                # Taper: linear peak → 0
                accel_profile[ramp_msgs + hold_msgs:] = np.linspace(peak_accel, 0, taper_msgs)

                # Estimate dt from timestamps
                dt = 0.1
                if len(attack_indices) > 1:
                    try:
                        t0 = pd.to_numeric(df.loc[attack_indices[0], time_col], errors='coerce')
                        t1 = pd.to_numeric(df.loc[attack_indices[1], time_col], errors='coerce')
                        if pd.notna(t0) and pd.notna(t1) and abs(t1 - t0) > 0:
                            dt = abs(t1 - t0)
                    except (ValueError, TypeError):
                        pass

                # Integrate acceleration → speed (forward Euler; v[i] is
                # speed at END of step i, matching broadcast-at-end semantics).
                speed_profile = np.cumsum(accel_profile) * dt

                # Integrate speed → cumulative displacement using KINEMATIC
                # equation per step, NOT rectangular sum of end-of-step speed.
                # fix: the v2 code used cumsum(speed_profile)*dt
                # which treated end-of-step speed as average speed, producing
                # position that was ~2× too large per step. This created a
                # detectable pos_vel_delta signature (single-feature audit AUROC
                # 0.906). The correct per-step displacement with constant
                # acceleration over the step is: Δx = v_prev·dt + ½·a·dt², where
                # v_prev is speed at the START of step i (= speed_profile[i-1]
                # for i>0, or 0 for i=0).
                v_prev = np.concatenate(([0.0], speed_profile[:-1]))
                step_disp = v_prev * dt + 0.5 * accel_profile * (dt ** 2)
                cum_displacement = np.cumsum(step_disp)

                # Get heading for position offset direction
                heading_rad = np.radians(df.loc[attack_indices[0], heading_col])

                for i, idx in enumerate(attack_indices):
                    df.loc[idx, accel_col] = accel_profile[i]
                    df.loc[idx, speed_col] = speed_profile[i]

                    # Co-spoof position along heading.
                    # Note: i=0 has speed_profile[0] ≈ accel[0]*dt (tiny, ~0.025 m/s)
                    # but no position offset yet. This one-message gap is negligible
                    # (~0.0025 m at 0.1s dt) and mimics real sensor report latency.
                    if i > 0:
                        offset_north = cum_displacement[i] * np.cos(heading_rad)
                        offset_east = cum_displacement[i] * np.sin(heading_rad)

                        lat = df.loc[idx, lat_col]
                        lon = df.loc[idx, lon_col]
                        new_lat, new_lon = self.offset_coordinates(lat, lon, offset_north, offset_east)
                        df.loc[idx, lat_col] = new_lat
                        df.loc[idx, lon_col] = new_lon

                    attack_mask.loc[idx] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        self.log_attack_summary(df, attack_mask)

        return df

    def _find_contiguous_segments(self, indices: list) -> list:
        """Find contiguous segments in a list of indices."""
        if not indices:
            return []

        segments = []
        current_segment = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]

        segments.append(current_segment)
        return segments
