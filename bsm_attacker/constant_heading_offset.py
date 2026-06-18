"""
Constant Heading Offset Attack (VASP heading/ConstantOffset)
============================================================

The attacker applies a fixed angular bias to the true heading for all
attacked messages of a given vehicle.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: h' = (h + offset) % 360, offset drawn once per vehicle
- Goal: Create consistent directional misinformation

Literature Reference:
- VASP: heading/ConstantOffset
- Fixed angular displacement from true heading
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class ConstantHeadingOffsetAttacker(BaseAttacker):
    """
    Implements the 'Constant Heading Offset' attack (VASP heading/ConstantOffset).

    Each attacked vehicle has a fixed angular offset applied to all its
    attacked messages, creating a systematically rotated heading.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Constant Heading Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # headingAttackOffset = π/6 rad ≈ 30° (Ansari 2023).
                      offset_range_deg: tuple = (30, 30),
                      bidirectional: bool = True,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Constant Heading Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            offset_range_deg: Tuple (min, max) angular offset in degrees
            bidirectional: If True, offset can be clockwise or counter-clockwise
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

            offset_deg = np.random.uniform(offset_range_deg[0], offset_range_deg[1])
            if bidirectional and np.random.random() < 0.5:
                offset_deg = -offset_deg

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            df.loc[attack_indices, heading_col] = (
                df.loc[attack_indices, heading_col] + offset_deg
            ) % 360
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Heading'] = original_headings

        self.log_attack_summary(df, attack_mask)
        print(f"Offset range: {offset_range_deg[0]}-{offset_range_deg[1]} deg (bidirectional={bidirectional})")
        return df
