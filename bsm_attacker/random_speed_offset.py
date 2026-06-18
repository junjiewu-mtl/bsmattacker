"""
Random Speed Offset Attack (VeReMi A8)
=======================================

The attacker applies a different random additive offset to the true
speed in each attacked message, creating noisy speed readings.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: v' = v + U(-k, +k) per message
- Goal: Create erratic but plausible speed variations

Literature Reference:
- VeReMi Extension: Random Speed Offset (A8)
- VASP: speed/RandomOffset — additive random offset to speed vector
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomSpeedOffsetAttacker(BaseAttacker):
    """
    Implements the 'Random Speed Offset' attack (VeReMi A8).

    Each attacked message receives an independently drawn additive
    speed offset, creating noisy but partially plausible speed readings.
    Matches VASP's additive mechanism.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Speed Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      max_offset_mps: float = 10.0,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Speed Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            max_offset_mps: Maximum additive offset in m/s (symmetric: [-k, +k])
            target_vehicles: List of Device_IDs to attack (None = all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        speed_col = 'Speed_mps' if 'Speed_mps' in df.columns else 'speed_mps'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        attack_mask = pd.Series(False, index=df.index)
        original_speeds = df[speed_col].copy()

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) == 0:
                continue

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            offsets = np.random.uniform(-max_offset_mps, max_offset_mps, size=n_attack)
            df.loc[attack_indices, speed_col] = (
                df.loc[attack_indices, speed_col] + offsets
            ).clip(lower=0)
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Speed'] = original_speeds

        self.log_attack_summary(df, attack_mask)
        print(f"Max offset: ±{max_offset_mps} m/s")
        return df
