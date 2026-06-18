"""
Base Attacker Class
===================

Provides the foundation for all attack implementations with common utilities
for coordinate manipulation, distance calculations, and attack labeling.

Supports both legacy and extended CSV column formats.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import warnings

# Suppress pandas SettingWithCopyWarning and FutureWarnings only
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*SettingWithCopy.*')


# Column mapping between legacy (22-col) and new (26-col) formats
# Key: canonical name used in attack code, Value: possible column names in CSV
COLUMN_ALIASES = {
    'device_id': ['Device_ID', 'device_id', 'station_id'],
    'timestamp': ['Tx_Timestamp', 'timestamp', 'Timestamp_ms', 'timestamp_s'],
    'latitude': ['Latitude_deg', 'latitude'],
    'longitude': ['Longitude_deg', 'longitude'],
    'speed': ['Speed_mps', 'speed_mps'],
    'heading': ['Heading_deg', 'heading_deg'],
    'accel_long': ['Accel_Long_mps2', 'accel_long_mps2', 'accel_long_ms2'],
    'accel_lat': ['Accel_Lat_mps2', 'accel_lat_mps2', 'accel_lat_ms2'],
    'accel_vert': ['Accel_Vert_mps2', 'accel_vert_mps2'],
    'yaw_rate': ['Yaw_Rate', 'yaw_rate_degs', 'yaw_rate', 'yaw_rate_dps'],
    'scenario_label': ['Scenario_Label', 'Scenario_Label_Raw'],
    'direction': ['direction'],  # Only in 26-col format
    'seq_num': ['seq_num'],  # Only in 26-col format
    'msg_count': ['Message_Count', 'msg_count'],
}


def detect_csv_format(df: pd.DataFrame) -> str:
    """
    Detect whether the DataFrame is in 22-column (legacy) or 26-column (new) format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        '26-col' for new format, '22-col' for legacy format
    """
    if 'seq_num' in df.columns:
        return '26-col'
    elif 'Device_ID' in df.columns or 'Latitude_deg' in df.columns:
        return '22-col'
    else:
        # Try to infer from column count
        return '26-col' if len(df.columns) >= 25 else '22-col'


def get_column_name(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """
    Get the actual column name in the DataFrame for a canonical column name.
    
    Args:
        df: Input DataFrame
        canonical_name: Canonical name (key in COLUMN_ALIASES)
        
    Returns:
        Actual column name in df, or None if not found
    """
    if canonical_name not in COLUMN_ALIASES:
        # Not in aliases, try direct match
        return canonical_name if canonical_name in df.columns else None
    
    for alias in COLUMN_ALIASES[canonical_name]:
        if alias in df.columns:
            return alias
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical format for consistent processing.
    Creates a copy with standardized column names.
    
    Args:
        df: Input DataFrame with either format
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    rename_map = {}
    
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename_map[alias] = canonical
                break
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


class BaseAttacker(ABC):
    """
    Abstract base class for all BSM attackers.
    
    All attack implementations must inherit from this class and implement
    the `inject_attack()` method.
    """
    
    def __init__(self, attack_name: str, random_seed: int = 42):
        """
        Initialize the base attacker.

        Args:
            attack_name: Human-readable name of the attack
            random_seed: Random seed for reproducibility
        """
        self.attack_name = attack_name
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        # Also set global seed for backwards compatibility with subclasses
        # that still use np.random directly (to be migrated incrementally)
        np.random.seed(random_seed)
        
    @abstractmethod
    def inject_attack(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Inject the attack into the dataset.
        
        This method must be implemented by all subclasses.
        
        Args:
            df: Input DataFrame with BSM data
            **kwargs: Attack-specific parameters
            
        Returns:
            DataFrame with injected attacks and attack labels
        """
        pass
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Coordinates of first point (degrees)
            lat2, lon2: Coordinates of second point (degrees)
            
        Returns:
            Distance in meters
        """
        # Earth radius in meters
        R = 6371000
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        # Haversine formula
        a = np.sin(delta_lat / 2) ** 2 + \
            np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    @staticmethod
    def offset_coordinates(lat: float, lon: float,
                          offset_north: float, offset_east: float) -> Tuple[float, float]:
        """
        Offset coordinates by a distance in metres.

        Detects whether (lat, lon) are in WGS-84 degrees or SUMO Cartesian
        metres (VeReMi-Ext synth) and applies the offset directly in metres
        when the corpus is Cartesian. Real-site corpora use the geodetic
        Earth-radius conversion.

        Args:
            lat: latitude — degrees for real sites, pos_y metres for synth
            lon: longitude — degrees for real sites, pos_x metres for synth
            offset_north, offset_east: offset in metres

        Returns:
            (new_lat, new_lon) in the same units as the input.
        """
        # Synth (VeReMi-Ext) lat/lon are SUMO local metres (max abs > 90);
        # real corpora are WGS-84 degrees (max abs ≤ 90).
        if abs(lat) > 90.0:
            return lat + offset_north, lon + offset_east
        R = 6371000  # Earth radius in metres
        new_lat = lat + (offset_north / R) * (180 / np.pi)
        new_lon = lon + (offset_east / R) * (180 / np.pi) / np.cos(lat * np.pi / 180)
        return new_lat, new_lon
    
    @staticmethod
    def add_attack_labels(df: pd.DataFrame, attack_mask: pd.Series, 
                         attack_type: str) -> pd.DataFrame:
        """
        Add attack labels to the DataFrame.
        
        Args:
            df: Input DataFrame
            attack_mask: Boolean mask indicating attacked messages
            attack_type: Name of the attack type
            
        Returns:
            DataFrame with added 'Attack_Label' and 'Is_Attack' columns
        """
        df['Is_Attack'] = attack_mask.astype(int)
        df['Attack_Label'] = 'Benign'
        df.loc[attack_mask, 'Attack_Label'] = attack_type
        
        return df
    
    def log_attack_summary(self, df: pd.DataFrame, attack_mask: pd.Series):
        """
        Print a summary of the attack injection.

        Args:
            df: DataFrame with injected attacks
            attack_mask: Boolean mask indicating attacked messages
        """
        if getattr(self, 'verbose', True) is False:
            return
        total_messages = len(df)
        attacked_messages = attack_mask.sum()
        attack_percentage = (attacked_messages / total_messages) * 100

        print(f"\n{'='*70}")
        print(f"Attack: {self.attack_name}")
        print(f"{'='*70}")
        print(f"Total Messages:    {total_messages:,}")
        print(f"Attacked Messages: {attacked_messages:,} ({attack_percentage:.2f}%)")
        print(f"Benign Messages:   {total_messages - attacked_messages:,} ({100 - attack_percentage:.2f}%)")
        print(f"{'='*70}\n")
