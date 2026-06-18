"""
Constant Heading Attack (VASP heading/Constant)
================================================

The attacker reports a fixed heading value regardless of actual vehicle
direction, as if the heading sensor is frozen.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: Replace heading with a constant value drawn once per vehicle
- Goal: Mislead heading-based plausibility checks and IMA applications

Literature Reference:
- VASP: heading/Constant
- Heading field frozen at a single value throughout the attack window
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantHeadingAttacker(BaseAttacker):
    """
    Implements the 'Constant Heading' attack (VASP heading/Constant).

    Attacked messages report a fixed heading value chosen once per vehicle,
    regardless of actual direction of travel.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Heading", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Heading' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            target_vehicles: List of Device_IDs to attack (None = all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        heading_col = 'Heading_deg' if 'Heading_deg' in df.columns else 'heading_deg'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        attack_mask = pd.Series(False, index=df.index)
        original_headings = df[heading_col].copy()

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_indices) == 0:
                continue

            frozen_heading = np.random.uniform(0, 360)

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, heading_col] = frozen_heading
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Heading'] = original_headings

        self.log_attack_summary(df, attack_mask)
        return df
