"""
Random Position Attack (VeReMi Type 4)
======================================

This attack reports random positions within a defined radius of the actual position,
making it difficult to track the vehicle's true trajectory.

Attack Mechanism:
- Target: All messages from selected vehicles
- Modification: Add random offset to latitude/longitude
- Goal: Confuse tracking systems, enable masking of true position

Literature Reference:
- VeReMi Dataset: Type 4 "Random Position" attack
- Each message has position offset by random (lat, lon) within radius
- Position jumps randomly, unlike smooth trajectory of legitimate vehicles

Use Case:
This attack enables position privacy violation or tracking evasion.
It can also be used to mask malicious driving behavior or confuse
intersection collision avoidance systems.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class RandomPositionAttacker(BaseAttacker):
    """
    Implements the 'Random Position' attack (VeReMi Type 4).
    
    The attacker reports positions with random offsets from true location,
    creating an erratic, untraceable movement pattern.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Random Position", random_seed=random_seed)
    
    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.2,
                     position_offset_meters: tuple = (5, 50),
                     consistent_bias: bool = False,
                     bias_direction_deg: float = None,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Random Position' attacks into BSM data.
        
        Attack Modes:
        1. Pure Random: Each message gets random offset in random direction
        2. Consistent Bias: All messages offset in same direction (harder to detect)
        
        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of messages to attack (0.0 to 1.0)
            position_offset_meters: Tuple (min, max) offset in meters
            consistent_bias: If True, offset in consistent direction (not truly random)
            bias_direction_deg: Direction of bias (0-360), random if None
            target_vehicles: List of Device_IDs to attack (None = attack all vehicles)
            
        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()
        
        # Determine column names (support both 22-col and 26-col formats)
        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        lat_col = 'Latitude_deg' if 'Latitude_deg' in df.columns else 'latitude'
        lon_col = 'Longitude_deg' if 'Longitude_deg' in df.columns else 'longitude'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'
        
        df = df.sort_values([device_col, time_col]).reset_index(drop=True)
        
        # Create attack columns
        attack_mask = pd.Series(False, index=df.index)
        original_lats = df[lat_col].copy()
        original_lons = df[lon_col].copy()
        
        if target_vehicles is None:
            target_vehicles = df[device_col].unique()
        
        from bsm_attacker.geo_constants import (
            DEG_LAT_TO_M, deg_lon_to_m, positions_are_metres,
        )
        if positions_are_metres(df[lat_col].to_numpy()):
            meters_per_degree_lat = 1.0
            meters_per_degree_lon = 1.0
        else:
            meters_per_degree_lat = DEG_LAT_TO_M
            meters_per_degree_lon = deg_lon_to_m(df[lat_col].median())
        
        for vehicle_id in target_vehicles:
            vehicle_mask = df[device_col] == vehicle_id
            vehicle_indices = df[vehicle_mask].index.tolist()
            
            if len(vehicle_indices) == 0:
                continue
            
            n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
            attack_indices = np.random.choice(vehicle_indices, size=n_attack, replace=False)
            
            if consistent_bias:
                if bias_direction_deg is not None:
                    direction = np.radians(bias_direction_deg)
                else:
                    direction = np.random.uniform(0, 2 * np.pi)
            
            for idx in attack_indices:
                offset_m = np.random.uniform(position_offset_meters[0], position_offset_meters[1])
                
                if not consistent_bias:
                    direction = np.random.uniform(0, 2 * np.pi)
                
                delta_lat = (offset_m * np.cos(direction)) / meters_per_degree_lat
                delta_lon = (offset_m * np.sin(direction)) / meters_per_degree_lon
                
                df.loc[idx, lat_col] = df.loc[idx, lat_col] + delta_lat
                df.loc[idx, lon_col] = df.loc[idx, lon_col] + delta_lon
                
                attack_mask.loc[idx] = True
        
        # Add attack labels
        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Latitude'] = original_lats
        df['Original_Longitude'] = original_lons
        
        # Print summary
        self.log_attack_summary(df, attack_mask)
        print(f"Position offset range: {position_offset_meters[0]}-{position_offset_meters[1]} meters")
        print(f"Consistent bias: {consistent_bias}")
        
        return df
