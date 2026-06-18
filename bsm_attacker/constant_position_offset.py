"""
Constant Position Offset Attack (VeReMi Type 2)
================================================

The attacker adds a fixed offset to the true position in every attacked
message, creating a consistently displaced "shadow" of the real vehicle.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Add constant (delta_lat, delta_lon) to true position
- Goal: Displace perceived position, fool collision avoidance

Literature Reference:
- VeReMi Dataset: Type 2 "Constant Position Offset"
- Lon_t = Lon_t + delta_Lon_c, Lat_t = Lat_t + delta_Lat_c
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantPositionOffsetAttacker(BaseAttacker):
    """
    Implements the 'Constant Position Offset' attack (VeReMi Type 2).

    Attacked messages have their position shifted by a fixed offset
    determined once per vehicle.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Position Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # posAttackOffset = 10 m (Ansari 2023). VASP's
                      # ConstantOffset adds a fixed 10 m offset; we use
                      # (10, 10) = exactly 10 m magnitude per attacker.
                      offset_meters: tuple = (10, 10),
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Position Offset' attacks into BSM data.

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

        from bsm_attacker.geo_constants import (
            DEG_LAT_TO_M, deg_lon_to_m, positions_are_metres,
        )
        # Synth (VeReMi-Ext) stores SUMO local metres in lat/lon; real sites
        # store WGS-84 degrees. Apply the deg→m conversion only for degrees.
        if positions_are_metres(df[lat_col].to_numpy()):
            meters_per_degree_lat = 1.0
            meters_per_degree_lon = 1.0
        else:
            meters_per_degree_lat = DEG_LAT_TO_M
            meters_per_degree_lon = deg_lon_to_m(df[lat_col].median())

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) == 0:
                continue

            offset_m = np.random.uniform(offset_meters[0], offset_meters[1])
            direction = np.random.uniform(0, 2 * np.pi)
            delta_lat = (offset_m * np.cos(direction)) / meters_per_degree_lat
            delta_lon = (offset_m * np.sin(direction)) / meters_per_degree_lon

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, lat_col] += delta_lat
            df.loc[attack_indices, lon_col] += delta_lon
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Latitude'] = original_lats
        df['Original_Longitude'] = original_lons

        self.log_attack_summary(df, attack_mask)
        print(f"Offset range: {offset_meters[0]}-{offset_meters[1]} meters")
        return df
