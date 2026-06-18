"""
Constant Speed Offset Attack (VeReMi A6)
=========================================

The attacker applies a fixed additive bias to the true speed value
for all attacked messages of a given vehicle.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: v' = v + offset, offset drawn once per vehicle
- Goal: Mislead speed-based plausibility checks with systematic bias

Literature Reference:
- VeReMi Extension: Constant Speed Offset (A6)
- VASP: speed/ConstantOffset — additive offset to speed vector
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantSpeedOffsetAttacker(BaseAttacker):
    """
    Implements the 'Constant Speed Offset' attack (VeReMi A6).

    Each attacked vehicle has a fixed additive speed bias applied
    to all its attacked messages, creating a systematically faster or
    slower reported speed. Matches VASP's additive mechanism.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Speed Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # speedAttackOffset = 10 m/s (Ansari 2023).
                      offset_range: tuple = (10.0, 10.0),
                      bidirectional: bool = True,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Speed Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            offset_range: Tuple (min, max) for the additive offset in m/s
            bidirectional: If True, offset can be positive or negative
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

            offset_factor = np.random.uniform(offset_range[0], offset_range[1])
            if bidirectional and np.random.random() < 0.5:
                offset_factor = -offset_factor

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, speed_col] = (
                df.loc[attack_indices, speed_col] + offset_factor
            ).clip(lower=0)
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Speed'] = original_speeds

        self.log_attack_summary(df, attack_mask)
        print(f"Offset range: {offset_range[0]}-{offset_range[1]} (bidirectional={bidirectional})")
        return df
