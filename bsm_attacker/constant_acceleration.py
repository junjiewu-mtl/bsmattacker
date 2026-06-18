"""
Constant Acceleration Attack (VASP acceleration/Constant)
=========================================================

The attacker reports fixed acceleration values regardless of actual
vehicle dynamics, as if the accelerometer is frozen.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Replace accel_long and accel_lat with constant values per vehicle
- Goal: Mislead acceleration-based plausibility checks

Literature Reference:
- VASP: acceleration/Constant
- Acceleration fields frozen at single values throughout the attack window
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantAccelerationAttacker(BaseAttacker):
    """
    Implements the 'Constant Acceleration' attack (VASP acceleration/Constant).

    Attacked messages report fixed longitudinal and lateral acceleration
    values chosen once per vehicle, regardless of actual dynamics.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Acceleration", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      accel_long_range: tuple = (-5, 5),
                      accel_lat_range: tuple = (-3, 3),
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Acceleration' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            accel_long_range: Tuple (min, max) for frozen longitudinal accel (m/s²)
            accel_lat_range: Tuple (min, max) for frozen lateral accel (m/s²)
            target_vehicles: List of Device_IDs to attack (None = all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        accel_long_col = 'Accel_Long_mps2' if 'Accel_Long_mps2' in df.columns else 'accel_long_ms2'
        accel_lat_col = 'Accel_Lat_mps2' if 'Accel_Lat_mps2' in df.columns else 'accel_lat_ms2'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        attack_mask = pd.Series(False, index=df.index)
        original_accel_long = df[accel_long_col].copy()
        original_accel_lat = df[accel_lat_col].copy()

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) == 0:
                continue

            frozen_long = np.random.uniform(accel_long_range[0], accel_long_range[1])
            frozen_lat = np.random.uniform(accel_lat_range[0], accel_lat_range[1])

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, accel_long_col] = frozen_long
            df.loc[attack_indices, accel_lat_col] = frozen_lat
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Accel_Long'] = original_accel_long
        df['Original_Accel_Lat'] = original_accel_lat

        self.log_attack_summary(df, attack_mask)
        print(f"Accel long range: {accel_long_range[0]} to {accel_long_range[1]} m/s²")
        print(f"Accel lat range: {accel_lat_range[0]} to {accel_lat_range[1]} m/s²")
        return df
