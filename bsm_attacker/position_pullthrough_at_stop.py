"""
Position Pullthrough At Stop Attack
===================================

This attack targets vehicles in STATIONARY scenarios (e.g., stopped at traffic lights).
The attacker falsely reports that they have already passed through the intersection,
potentially causing confusion or dangerous reactions from other vehicles.

Attack Mechanism:
- Target: Messages with Scenario_Label = 'Stationary_Wait' (traffic light waits, 5–60 s)
- Modification: Offset position forward by 20-50 meters (simulating passing through intersection)
- Realism: Attacks contiguous sequences with consistent offset + GPS noise
- Co-spoofing: Adds a 3–5 message speed bell curve (0 → 1.5–3.0 m/s → 0) to
  simulate "pulled forward and stopped", sets heading toward the offset
  direction, and spoofs acceleration consistent with the speed profile so the
  attack avoids trivial position-velocity consistency detection.

Literature Reference:
- VeReMi "Constant Position" attack type, but scenario-aware
- Key improvement: Add GPS noise to spoofed position (2-5m std) for realism
- Attack contiguous message sequences (not random) to mimic GPS lock behavior

Use Case:
This attack is particularly dangerous at intersections where vehicles rely on BSM data
to coordinate right-of-way and collision avoidance.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class PositionPullthroughAtStopAttacker(BaseAttacker):
    """
    Implements the 'Position Pullthrough At Stop' attack for stationary scenarios.

    Attacks contiguous sequences with consistent base offset plus realistic
    GPS noise, mimicking real GPS lock behavior.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Position Pullthrough At Stop", random_seed=random_seed)
    
    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.3,
                     offset_range: tuple = (20, 50),
                     gps_noise_std: float = 2.5,
                     speed_ramp_msgs: int = 5,
                     peak_speed_range: tuple = (1.5, 3.0),
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Position Pullthrough At Stop' attacks into Stationary_Wait
        messages (traffic light waits, 5–60 s duration); brief stops and
        parking segments are not targeted.

        Co-spoofs position, speed, heading, and acceleration to defeat
        position-velocity consistency detection. Adds a bell-curve speed ramp
        simulating "pulled forward and stopped back".

        Args:
            df: Input DataFrame with BSM data (must have 'Scenario_Label' column)
            attack_ratio: Proportion of stationary messages to attack (0.0 to 1.0)
            offset_range: Tuple of (min_offset, max_offset) in meters for position spoofing
            gps_noise_std: Standard deviation of GPS noise added to spoofed position (meters)
            speed_ramp_msgs: Number of messages for the speed bell curve (3–5)
            peak_speed_range: Tuple of (min, max) peak speed in m/s for the ramp
            target_vehicles: List of Device_IDs to attack (None = attack all vehicles)

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

        n_wait = (df[scenario_col] == 'Stationary_Wait').sum()
        n_flat = (df[scenario_col] == 'Stationary').sum()
        if n_wait == 0 and n_flat > 0:
            import warnings
            warnings.warn(
                f"position_pullthrough_at_stop: 0 Stationary_Wait rows but {n_flat} flat "
                f"'Stationary' rows found. Did you run the "
                f"duration-aware labeling? Attack will inject 0 messages.",
                stacklevel=2,
            )

        # 5-60s stops are red-light waits at SIGNALIZED intersections
        # (stop signs produce <5s stops = Stationary_Brief). In deployment the
        # signalized bit is derived from J2735 MAP signal group ID per approach
        # (C2CCC Automotive Requirements for SPaT and MAP v1.5.0). Offline we
        # proxy via OSM highway=traffic_signals node proximity. VeReMi synthetic
        # (no OSM, no MAP) falls back to kinematic-only Stationary_Wait.
        if 'is_signalized' in df.columns:
            context_ok = df['is_signalized'].fillna(False).astype(bool)
        elif 'in_intersection_zone' in df.columns:
            context_ok = df['in_intersection_zone'].fillna(False).astype(bool)
        else:
            context_ok = pd.Series(True, index=df.index)

        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (df[scenario_col] == 'Stationary_Wait') & context_ok
            vehicle_stationary_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_stationary_indices) == 0:
                continue

            num_to_attack = max(1, int(len(vehicle_stationary_indices) * attack_ratio))
            segments = self._find_contiguous_segments(vehicle_stationary_indices)

            attacked_indices = []
            remaining = num_to_attack

            for segment in segments:
                if remaining <= 0:
                    break
                segment_attack_count = min(len(segment), max(1, int(len(segment) * attack_ratio)))
                segment_attack_count = min(segment_attack_count, remaining)
                attacked_indices.extend(segment[:segment_attack_count])
                remaining -= segment_attack_count

            if not attacked_indices:
                continue

            base_offset_distance = np.random.uniform(offset_range[0], offset_range[1])
            peak_speed = np.random.uniform(peak_speed_range[0], peak_speed_range[1])

            # v2.0: Build speed bell curve for the first N messages
            # Shape: 0 → peak → 0 over speed_ramp_msgs messages
            n_ramp = min(speed_ramp_msgs, len(attacked_indices))
            ramp_profile = np.sin(np.linspace(0, np.pi, n_ramp))  # bell curve [0..1..0]
            speed_profile = ramp_profile * peak_speed

            # Estimate dt from timestamps
            dt = 0.1
            if len(attacked_indices) > 1:
                try:
                    t0 = pd.to_numeric(df.loc[attacked_indices[0], time_col], errors='coerce')
                    t1 = pd.to_numeric(df.loc[attacked_indices[1], time_col], errors='coerce')
                    if pd.notna(t0) and pd.notna(t1) and abs(t1 - t0) > 0:
                        dt = abs(t1 - t0)
                except (ValueError, TypeError):
                    pass

            # Compute acceleration from speed profile (forward difference)
            accel_profile = np.zeros(n_ramp)
            for k in range(1, n_ramp):
                accel_profile[k] = (speed_profile[k] - speed_profile[k - 1]) / dt

            # Compute cumulative position offset from speed profile
            # Each message contributes speed * dt metres in the heading direction
            cum_position = np.cumsum(speed_profile * dt)
            # Scale so final cumulative matches base_offset_distance
            if cum_position[-1] > 0:
                position_scale = base_offset_distance / cum_position[-1]
            else:
                position_scale = 1.0

            # Heading toward offset direction (original heading ± small noise)
            offset_heading = df.loc[attacked_indices[0], heading_col]

            for i, idx in enumerate(attacked_indices):
                heading = df.loc[idx, heading_col]

                if i < n_ramp:
                    # Ramp phase: co-spoof speed + position + acceleration + heading
                    current_offset = cum_position[i] * position_scale
                    df.loc[idx, speed_col] = speed_profile[i]
                    df.loc[idx, accel_col] = accel_profile[i]
                    # Tilt heading toward offset direction
                    df.loc[idx, heading_col] = offset_heading + np.random.normal(0, 2.0)
                else:
                    # Hold phase: position at full offset, speed back to 0
                    current_offset = base_offset_distance
                    df.loc[idx, speed_col] = 0.0
                    df.loc[idx, accel_col] = 0.0

                noise_north = np.random.normal(0, gps_noise_std)
                noise_east = np.random.normal(0, gps_noise_std)

                offset_north = current_offset * np.cos(np.radians(offset_heading)) + noise_north
                offset_east = current_offset * np.sin(np.radians(offset_heading)) + noise_east

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
        """
        Find contiguous segments in a list of indices.
        
        Returns list of lists, where each inner list is a contiguous segment.
        """
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
