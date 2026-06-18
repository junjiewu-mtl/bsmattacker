"""
Ghost Vehicle Attack (VASP position/ghost_vehicle)
===================================================

The attacker fabricates entirely new vehicle identities by replaying
real BSM trajectories with fresh device IDs and shifted positions,
creating phantom vehicles in the network.

Attack Mechanism:
- Target: Complete trajectories from real vehicles
- Modification: Clone trajectory with new ID + spatial offset
- Goal: Inflate perceived traffic density and mislead cooperative systems

Literature Reference:
- VASP: position/ghost_vehicle
- Creates phantom vehicles that don't physically exist
- Distinguished from Data Replay by primary goal of identity fabrication
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class GhostVehicleAttacker(BaseAttacker):
    """
    Implements the 'Ghost Vehicle' attack (VASP position/ghost_vehicle).

    Clones real vehicle trajectories with new device IDs and spatial
    offsets, creating phantom vehicles that appear in the network.
    """

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Ghost Vehicle", random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.2,
                      n_ghosts: int = None,
                      position_shift_range: tuple = (10, 50),
                      speed_jitter: float = 0.0,
                      target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Ghost Vehicle' attacks into BSM data.

        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Fraction of unique vehicles to clone as ghosts (0.0–1.0).
                         Used to compute n_ghosts when n_ghosts is not provided.
            n_ghosts: Number of ghost vehicles to create (overrides attack_ratio if set)
            position_shift_range: Tuple (min, max) meters to offset ghost position
            speed_jitter: Proportional speed wobble; default 0.0 for
                         attacker-OBU realism (no self-jitter; a real adversary
                         in full control of transmitted fields would not add a
                         5% wobble that creates a kinematic fingerprint).
            target_vehicles: List of Device_IDs to clone from (None = random selection)

        Returns:
            DataFrame with original + ghost vehicle messages
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        lat_col = 'Latitude_deg' if 'Latitude_deg' in df.columns else 'latitude'
        lon_col = 'Longitude_deg' if 'Longitude_deg' in df.columns else 'longitude'
        speed_col = 'Speed_mps' if 'Speed_mps' in df.columns else 'speed_mps'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        # Mark original messages as normal
        df['Is_Attack'] = 0
        df['Attack_Label'] = 'Benign'

        all_vehicles = df[device_col].unique()

        # Derive n_ghosts from attack_ratio if not explicitly set
        if n_ghosts is None:
            n_ghosts = max(1, int(len(all_vehicles) * attack_ratio))

        if target_vehicles is None:
            n_sources = min(n_ghosts, len(all_vehicles))
            source_vehicles = np.random.choice(all_vehicles, size=n_sources, replace=False)
        else:
            source_vehicles = target_vehicles[:n_ghosts]

        from bsm_attacker.geo_constants import (
            DEG_LAT_TO_M, deg_lon_to_m, positions_are_metres,
        )
        if positions_are_metres(df[lat_col].to_numpy()):
            meters_per_deg_lat = 1.0
            meters_per_deg_lon = 1.0
        else:
            meters_per_deg_lat = DEG_LAT_TO_M
            meters_per_deg_lon = deg_lon_to_m(df[lat_col].median())

        ghost_sequences = []
        for i, source_id in enumerate(source_vehicles):
            source_df = df[df[device_col] == source_id].copy()
            if len(source_df) == 0:
                continue

            ghost_id = f"GHOST_{i + 1:03d}"

            # Random spatial offset
            shift_m = np.random.uniform(position_shift_range[0], position_shift_range[1])
            direction = np.random.uniform(0, 2 * np.pi)
            delta_lat = (shift_m * np.cos(direction)) / meters_per_deg_lat
            delta_lon = (shift_m * np.sin(direction)) / meters_per_deg_lon

            ghost_df = source_df.copy()
            ghost_df[device_col] = ghost_id
            ghost_df[lat_col] = ghost_df[lat_col] + delta_lat
            ghost_df[lon_col] = ghost_df[lon_col] + delta_lon

            # Add speed jitter to make ghost less identical to source
            if speed_jitter > 0 and speed_col in ghost_df.columns:
                jitter = np.random.uniform(
                    1 - speed_jitter, 1 + speed_jitter, size=len(ghost_df)
                )
                ghost_df[speed_col] = (ghost_df[speed_col] * jitter).clip(lower=0)

            ghost_df['Is_Attack'] = 1
            ghost_df['Attack_Label'] = 'Ghost Vehicle'
            ghost_df['Ghost_Source'] = str(source_id)

            ghost_sequences.append(ghost_df)

        if ghost_sequences:
            ghost_all = pd.concat(ghost_sequences, ignore_index=True)
            df = pd.concat([df, ghost_all], ignore_index=True)
            df = df.sort_values(time_col).reset_index(drop=True)

        if 'Ghost_Source' not in df.columns:
            df['Ghost_Source'] = ''
        else:
            df['Ghost_Source'] = df['Ghost_Source'].fillna('')

        attack_mask = df['Is_Attack'] == 1
        self.log_attack_summary(df, attack_mask)
        print(f"Ghost vehicles created: {len(ghost_sequences)}")
        print(f"Position shift: {position_shift_range[0]}-{position_shift_range[1]} m")
        return df
