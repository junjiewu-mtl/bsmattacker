"""
Random Speed Attack (VeReMi Speed Malfunction)
===============================================

The attacker reports a freshly randomised speed value in each attacked
message, creating erratic speed readings unrelated to actual dynamics.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Replace speed with U(min, max) drawn independently per message
- Goal: Confuse speed-based plausibility checks and platooning logic

Literature Reference:
- VeReMi Extension: Speed malfunction - Random
- V_t = U([V_min, V_max]) each timestep
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomSpeedAttacker(BaseAttacker):
    """
    Implements the 'Random Speed' attack (VeReMi speed malfunction).

    Each attacked message receives an independently drawn random speed
    value, creating physically implausible speed jumps.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Speed", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      speed_range: tuple = (0, 40),
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Speed' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            speed_range: Tuple (min, max) speed in m/s for the random draw
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

            random_speeds = np.random.uniform(speed_range[0], speed_range[1], size=n_attack)
            df.loc[attack_indices, speed_col] = random_speeds
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Speed'] = original_speeds

        self.log_attack_summary(df, attack_mask)
        print(f"Speed range: {speed_range[0]}-{speed_range[1]} m/s")
        return df
