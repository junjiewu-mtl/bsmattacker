"""
Heading Lock Attack (v1.0)
===========================

This attack targets vehicles in TURNING scenarios and locks the heading to
the value at turn entry, making the vehicle appear to drive straight through
an intersection while it is actually turning.

Attack Mechanism:
- Target: Messages with Scenario_Label = 'Turning'
- Modification: Freeze heading at the value recorded at the start of the turn
- Realism: Co-spoofs yaw rate to ~0 and adjusts position along the locked
  heading to maintain heading/position consistency

Co-spoofing (multi-field consistency):
- Heading: locked to turn-entry value
- Yaw rate: set to ~0 deg/s (consistent with straight driving)
- Position: projected forward along locked heading using real speed,
  diverging from actual turn arc
- Speed: preserved from real data (speed is plausible for straight driving)

Literature Reference:
- VASP heading/Constant attack family — but scenario-aware (turning only)
- Key improvement: context-aware targeting of turns only (not random messages),
  making the attack semantically meaningful for collision avoidance

Use Case:
Intersection collision avoidance (ICA) systems rely on heading to predict
vehicle path. A heading-locked BSM during a turn makes the vehicle appear
on a collision course with cross-traffic, triggering false ICA alerts, or
conversely hides the actual turn from vehicles that need to yield.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


def _intersection_gate(df):
    """Restrict heading-lock to intersection turns (not rural curves).

    Deployment: J2735 MAP IntersectionGeometry polygon containment. Offline:
    OSM in_intersection_zone. Synthetic VeReMi lacks both, so no gate.
    """
    if 'in_intersection_zone' in df.columns:
        return df['in_intersection_zone'].fillna(False).astype(bool)
    import pandas as _pd
    return _pd.Series(True, index=df.index)


class HeadingLockAttacker(BaseAttacker):
    """
    Implements the 'Heading Lock' attack for turning scenarios.

    Locks heading at turn-entry value and co-spoofs yaw rate and position
    to maintain consistency with straight-line driving.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Heading Lock", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.5,
                     yaw_noise_std: float = 0.5,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Heading Lock' attacks into turning messages.

        Freezes heading at turn-entry value. Co-spoofs yaw rate to ~0
        and projects position along the locked heading.

        Args:
            df: Input DataFrame with BSM data (must have 'Scenario_Label' column)
            attack_ratio: Proportion of turning messages to attack (0.0 to 1.0)
            yaw_noise_std: Std dev of yaw rate noise in deg/s (small, to mimic
                           straight-line sensor noise rather than exact 0.0)
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
        yaw_col = get_column_name(df, 'yaw_rate') or 'Yaw_Rate'
        scenario_col = get_column_name(df, 'scenario_label') or 'Scenario_Label'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        if scenario_col not in df.columns:
            raise ValueError(f"DataFrame must have '{scenario_col}' column. Run scenario labeling first.")

        attack_mask = pd.Series(False, index=df.index)

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        context_ok = _intersection_gate(df)

        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (df[scenario_col] == 'Turning') & context_ok
            vehicle_turning_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_turning_indices) == 0:
                continue

            segments = self._find_contiguous_segments(vehicle_turning_indices)

            for segment in segments:
                num_to_attack = max(1, int(len(segment) * attack_ratio))
                attack_indices = segment[:num_to_attack]

                if not attack_indices:
                    continue

                # Lock heading to value at turn entry
                locked_heading = df.loc[attack_indices[0], heading_col]
                locked_heading_rad = np.radians(locked_heading)

                # Estimate dt
                dt = 0.1
                if len(attack_indices) > 1:
                    try:
                        t0 = pd.to_numeric(df.loc[attack_indices[0], time_col], errors='coerce')
                        t1 = pd.to_numeric(df.loc[attack_indices[1], time_col], errors='coerce')
                        if pd.notna(t0) and pd.notna(t1) and abs(t1 - t0) > 0:
                            dt = abs(t1 - t0)
                    except (ValueError, TypeError):
                        pass

                # Compute cumulative displacement along locked heading
                # using real speed values (speed is not spoofed)
                cum_disp = 0.0
                base_lat = df.loc[attack_indices[0], lat_col]
                base_lon = df.loc[attack_indices[0], lon_col]

                for i, idx in enumerate(attack_indices):
                    # Lock heading
                    df.loc[idx, heading_col] = locked_heading

                    # Co-spoof yaw rate to ~0 (straight-line noise)
                    if yaw_col in df.columns:
                        df.loc[idx, yaw_col] = np.random.normal(0, yaw_noise_std)

                    # Co-spoof position: project along locked heading
                    if i > 0:
                        speed = df.loc[idx, speed_col]
                        if pd.notna(speed):
                            cum_disp += speed * dt

                        offset_north = cum_disp * np.cos(locked_heading_rad)
                        offset_east = cum_disp * np.sin(locked_heading_rad)

                        new_lat, new_lon = self.offset_coordinates(
                            base_lat, base_lon, offset_north, offset_east
                        )
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
