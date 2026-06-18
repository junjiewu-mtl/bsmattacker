"""
Slow Position Drift Attack
==========================

Safety-impacting navigation drift targeting FCW by creating meaningful
position error that biases TTC computation.

Attack Mechanism:
- Target: Cruising vehicles (FCW-relevant scenario)
- Position: AR(1) autocorrelated drift, 1-3 m/msg, max 30m cumulative
- Speed co-spoof: Bias speed by ±(0.5-2.0) m/s to maintain plausibility
  (position change roughly matches speed * dt)
- Safety impact: FCW computes TTC from position + speed. A 20-30m position
  error with matching speed bias causes FCW to under/over-estimate gap by
  1-3 seconds, enough to suppress or trigger false warnings.

Literature Reference:
- Zhu et al. (2022), "A Slowly Varying Spoofing Algorithm on Loosely Coupled
  GNSS/IMU Avoiding Multiple Anti-Spoofing Techniques", MDPI Sensors
- Dasgupta et al. (2024), "Unveiling the Stealthy Threat: Analyzing Slow Drift
  GPS Spoofing Attacks", arXiv:2401.01394

Safety Application Target: FCW (Forward Collision Warning)
- TTC = gap_distance / closing_speed
- 20-30m position error → 1-3s TTC bias → missed or false FCW alerts
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class SlowPositionDriftAttacker(BaseAttacker):
    """
    Implements the 'Slow Position Drift' attack.

    FCW-targeted with speed co-spoofing for position-velocity consistency.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Slow Position Drift", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.3,
                     drift_per_message: tuple = (1.0, 3.0),
                     max_cumulative_drift: float = 30.0,
                     ar1_rho: float = 0.95,
                     direction_innovation_std: float = 10.0,
                     speed_bias_range: tuple = (0.5, 2.0),
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Slow Position Drift' attacks with position drift + speed co-spoofing.

        Drift rate 1-3 m/msg (FCW-relevant), max 30m cumulative. Speed
        co-spoofed to maintain plausibility.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages to attack (0.0 to 1.0)
            drift_per_message: Tuple of (min, max) drift in metres per message
            max_cumulative_drift: Maximum total drift in metres before reset
            ar1_rho: AR(1) autocorrelation coefficient for drift direction
            direction_innovation_std: Std dev of AR(1) innovation noise (degrees)
            speed_bias_range: Tuple of (min, max) speed bias in m/s
            target_vehicles: List of Device_IDs to attack (None = attack all)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        time_col = get_column_name(df, 'timestamp') or 'Tx_Timestamp'
        lat_col = get_column_name(df, 'latitude') or 'Latitude_deg'
        lon_col = get_column_name(df, 'longitude') or 'Longitude_deg'
        speed_col = get_column_name(df, 'speed') or 'Speed_mps'
        scenario_col = get_column_name(df, 'scenario_label')

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        attack_mask = pd.Series(False, index=df.index)

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            if scenario_col and scenario_col in df.columns:
                # FCW-targeted v3.0: restrict spoofing to cruising rows.
                vehicle_mask = vehicle_mask & (df[scenario_col] == 'Cruising')
            vehicle_indices = df[vehicle_mask].index.tolist()
            if len(vehicle_indices) == 0:
                continue

            num_messages_to_attack = max(1, int(len(vehicle_indices) * attack_ratio))

            if len(vehicle_indices) > num_messages_to_attack:
                start_idx = np.random.randint(0, len(vehicle_indices) - num_messages_to_attack + 1)
                attack_indices = vehicle_indices[start_idx:start_idx + num_messages_to_attack]
            else:
                attack_indices = vehicle_indices

            n_attack = len(attack_indices)
            drift_distances = np.random.uniform(
                drift_per_message[0], drift_per_message[1], size=n_attack
            )

            # AR(1) autocorrelated drift direction
            drift_angles = np.empty(n_attack)
            drift_angles[0] = np.random.uniform(0, 360)
            for k in range(1, n_attack):
                innovation = np.random.normal(0, direction_innovation_std)
                drift_angles[k] = ar1_rho * drift_angles[k - 1] + (1 - ar1_rho) * innovation

            drift_north = drift_distances * np.cos(np.radians(drift_angles))
            drift_east = drift_distances * np.sin(np.radians(drift_angles))

            # Accumulate with max-drift resets
            cum_north = np.empty(n_attack)
            cum_east = np.empty(n_attack)
            cn, ce = 0.0, 0.0
            for k in range(n_attack):
                cn += drift_north[k]
                ce += drift_east[k]
                if np.sqrt(cn ** 2 + ce ** 2) > max_cumulative_drift:
                    drift_angles[k] = np.random.uniform(0, 360)
                    drift_north[k] = drift_distances[k] * np.cos(np.radians(drift_angles[k]))
                    drift_east[k] = drift_distances[k] * np.sin(np.radians(drift_angles[k]))
                    cn, ce = drift_north[k], drift_east[k]
                cum_north[k] = cn
                cum_east[k] = ce

            # Apply position drift
            lats = df.loc[attack_indices, lat_col].values
            lons = df.loc[attack_indices, lon_col].values
            from bsm_attacker.geo_constants import DEG_LAT_TO_M, positions_are_metres
            if positions_are_metres(df[lat_col].to_numpy()):
                # Synth (VeReMi-Ext): lat/lon are SUMO local metres; drift
                # values (cum_north, cum_east) are also metres → no conversion.
                new_lats = lats + cum_north
                new_lons = lons + cum_east
            else:
                new_lats = lats + cum_north / DEG_LAT_TO_M
                new_lons = lons + cum_east / (DEG_LAT_TO_M * np.cos(np.radians(lats)))

            df.loc[attack_indices, lat_col] = new_lats
            df.loc[attack_indices, lon_col] = new_lons

            # v3.0: Speed co-spoofing to maintain position-velocity consistency
            # Bias speed in the drift direction to make position change plausible
            if speed_col in df.columns:
                speed_bias = np.random.uniform(
                    speed_bias_range[0], speed_bias_range[1], size=n_attack
                )
                # Sign matches drift direction (positive = drifting forward)
                drift_magnitude = np.sqrt(cum_north ** 2 + cum_east ** 2)
                # Gradually increase speed bias as drift accumulates
                bias_scale = np.clip(drift_magnitude / max_cumulative_drift, 0, 1)
                # Alternate sign based on whether drift is increasing or decreasing
                speed_direction = np.sign(np.diff(drift_magnitude, prepend=0))
                speed_direction[speed_direction == 0] = 1
                df.loc[attack_indices, speed_col] += speed_bias * bias_scale * speed_direction

            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        self.log_attack_summary(df, attack_mask)

        return df
