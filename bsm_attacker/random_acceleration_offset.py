"""
Random Acceleration Offset Attack (VASP acceleration/RandomOffset)
==================================================================

The attacker perturbs the true acceleration by a different random
offset in each attacked message.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: a' = a + U(-k, +k) per message for both axes
- Goal: Create noisy but partially plausible acceleration deviations

Literature Reference:
- VASP: acceleration/RandomOffset
- Per-message random perturbation of true acceleration
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomAccelerationOffsetAttacker(BaseAttacker):
    """
    Implements the 'Random Acceleration Offset' attack (VASP acceleration/RandomOffset).

    Each attacked message receives independently drawn offsets added to
    the true longitudinal and lateral acceleration values.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Acceleration Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # accelerationAttackOffset = 2 m/s² (Ansari 2023).
                      # VASP has scalar acceleration; we apply ±2 to
                      # both longitudinal and lateral channels of the
                      # BSM Part-I 2D acceleration vector.
                      max_offset_long: float = 2.0,
                      max_offset_lat: float = 2.0,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Acceleration Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            max_offset_long: Max longitudinal offset in m/s² (symmetric: [-k, +k])
            max_offset_lat: Max lateral offset in m/s² (symmetric: [-k, +k])
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

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            offsets_long = np.random.uniform(-max_offset_long, max_offset_long, size=n_attack)
            offsets_lat = np.random.uniform(-max_offset_lat, max_offset_lat, size=n_attack)
            df.loc[attack_indices, accel_long_col] = (
                df.loc[attack_indices, accel_long_col] + offsets_long
            )
            df.loc[attack_indices, accel_lat_col] = (
                df.loc[attack_indices, accel_lat_col] + offsets_lat
            )
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Accel_Long'] = original_accel_long
        df['Original_Accel_Lat'] = original_accel_lat

        self.log_attack_summary(df, attack_mask)
        print(f"Max offset: long ±{max_offset_long} m/s², lat ±{max_offset_lat} m/s²")
        return df
