"""
Random Heading Offset Attack (VASP heading/RandomOffset)
========================================================

The attacker perturbs the true heading by a different random offset
in each attacked message.

Attack Mechanism:
- Target: Selected messages from all vehicles
- Modification: h' = (h + U(-k, +k)) % 360 per message
- Goal: Create noisy but partially plausible heading deviations

Literature Reference:
- VASP: heading/RandomOffset
- Per-message random perturbation of true heading
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomHeadingOffsetAttacker(BaseAttacker):
    """
    Implements the 'Random Heading Offset' attack (VASP heading/RandomOffset).

    Each attacked message receives an independently drawn angular offset
    from the true heading, creating noisy direction readings.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Heading Offset", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.15,
                      # match VASP CarApp.ned default
                      # headingAttackOffset = π/6 rad ≈ 30° (Ansari 2023).
                      max_offset_deg: float = 30,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Heading Offset' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages per vehicle to attack (0.0 to 1.0)
            max_offset_deg: Maximum angular offset in degrees (symmetric: [-k, +k])
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

            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)

            offsets = np.random.uniform(-max_offset_deg, max_offset_deg, size=n_attack)
            df.loc[attack_indices, heading_col] = (
                df.loc[attack_indices, heading_col] + offsets
            ) % 360
            attack_mask.loc[attack_indices] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Heading'] = original_headings

        self.log_attack_summary(df, attack_mask)
        print(f"Max offset: ±{max_offset_deg} deg")
        return df
