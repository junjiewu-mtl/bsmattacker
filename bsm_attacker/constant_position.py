"""
Constant Position Attack (VeReMi Type 1)
========================================

The attacker reports a fixed position regardless of actual movement,
as if the GPS receiver is stuck at a single coordinate.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Freeze lat/lon to the vehicle's first observed position
- Goal: Create phantom stationary vehicles, disrupt tracking

Literature Reference:
- VeReMi Dataset: Type 1 "Constant Position"
- Lon_t = Lon_c, Lat_t = Lat_c (constant throughout)
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantPositionAttacker(BaseAttacker):
    """
    Implements the 'Constant Position' attack (VeReMi Type 1).

    Attacked messages report the vehicle's first observed lat/lon,
    regardless of actual movement.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Position", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Position' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
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

            frozen_lat = df.loc[vehicle_indices[0], lat_col]
            frozen_lon = df.loc[vehicle_indices[0], lon_col]

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, lat_col] = frozen_lat
            df.loc[attack_indices, lon_col] = frozen_lon
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Latitude'] = original_lats
        df['Original_Longitude'] = original_lons

        self.log_attack_summary(df, attack_mask)
        return df
