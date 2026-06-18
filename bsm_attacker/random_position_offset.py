"""
Random Position Offset Attack (VeReMi Type 4)
==============================================

The attacker adds a *different* random offset to the true position in
each attacked message, creating a noisy trajectory around the real path.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Add random (delta_lat, delta_lon) drawn each timestep
- Goal: Make trajectory appear jittery, confuse plausibility checks

Literature Reference:
- VeReMi Dataset: Type 4 "Random Position Offset"
- Lon_t = Lon_t + U([-Lon_c, Lon_c])
- Lat_t = Lat_t + U([-Lat_c, Lat_c])
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomPositionOffsetAttacker(BaseAttacker):
    """
    Implements the 'Random Position Offset' attack (VeReMi Type 4).

    Each attacked message has a freshly drawn random offset added to
    the true position.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Position Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # posAttackOffset = 10 m (Ansari 2023). VASP applies
                      # U(-10, +10) per axis; we apply uniform magnitude
                      # in [10, 10] = exactly 10 m to remain consistent
                      # with VASP's scalar-bound semantics.
                      offset_meters: tuple = (10, 10),
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Position Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            offset_meters: Tuple (min, max) offset magnitude in meters
            target_vehicles: List of Device_IDs to attack (None = all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        lat_col = 'Latitude_deg' if 'Latitude_deg' in df.columns else 'latitude'
        lon_col = 'Longitude_deg' if 'Longitude_deg' in df.columns else 'longitude'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        attack_mask = pd.Series(False, index=df.index)
        original_lats = df[lat_col].copy()
        original_lons = df[lon_col].copy()

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) == 0:
                continue

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            offsets_m = np.random.uniform(offset_meters[0], offset_meters[1], size=n_attack)
            directions = np.random.uniform(0, 2 * np.pi, size=n_attack)

            north_m = offsets_m * np.cos(directions)
            east_m = offsets_m * np.sin(directions)

            for i, idx in enumerate(attack_indices):
                new_lat, new_lon = self.offset_coordinates(
                    df.loc[idx, lat_col], df.loc[idx, lon_col],
                    north_m[i], east_m[i]
                )
                df.loc[idx, lat_col] = new_lat
                df.loc[idx, lon_col] = new_lon

            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Latitude'] = original_lats
        df['Original_Longitude'] = original_lons

        self.log_attack_summary(df, attack_mask)
        print(f"Offset range: {offset_meters[0]}-{offset_meters[1]} meters")
        return df
