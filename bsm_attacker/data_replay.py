"""
Data Replay Attack (VeReMi Extension)
=====================================

This attack replays previously recorded BSM data at a later time,
creating "ghost" vehicles or duplicating vehicle trajectories.

Attack Mechanism:
- Target: Historical BSM sequences
- Modification: Replay old messages with updated timestamps
- Goal: Create confusion about actual traffic conditions

Literature Reference:
- VeReMi Extension: Data Replay attacks
- Replays valid BSM sequences from earlier time
- Creates phantom vehicles on the road network

Use Case:
This attack can create traffic congestion by making it appear there are
more vehicles than actually present. It can also be used for:
- Disrupting cooperative systems
- Creating false traffic patterns for routing manipulation
- Masking actual vehicle movements
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class DataReplayAttacker(BaseAttacker):
    """
    Implements the 'Data Replay' attack.
    
    The attacker captures BSM sequences and replays them at a later time,
    creating duplicate/ghost vehicles in the network.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Data Replay", random_seed=random_seed)
    
    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.15,
                     replay_delay_ms: tuple = (5000, 30000),
                     replay_sequence_length: tuple = (20, 100),
                     modify_device_id: bool = True,
                     position_shift_meters: float = 0,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Data Replay' attacks into BSM data.
        
        Attack Process:
        1. Select contiguous message sequences from source vehicles
        2. Copy sequences with modified timestamps (delayed)
        3. Optionally modify device ID to create "new" ghost vehicles
        4. Optionally shift position to create parallel ghost
        
        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of total messages to create as replays (0.0 to 1.0)
            replay_delay_ms: Tuple (min, max) delay in milliseconds for replay
            replay_sequence_length: Tuple (min, max) number of messages per replay
            modify_device_id: If True, assign new device ID (ghost vehicle)
                             If False, duplicate same ID (detectable by sequence analysis)
            position_shift_meters: Shift replayed positions by this distance (0 = same path)
            target_vehicles: List of Device_IDs to replay from (None = all vehicles)
            
        Returns:
            DataFrame with injected replay attacks (original + replayed messages)
        """
        df = df.copy()
        
        # Determine column names (support both 22-col and 26-col formats)
        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        lat_col = 'Latitude_deg' if 'Latitude_deg' in df.columns else 'latitude'
        lon_col = 'Longitude_deg' if 'Longitude_deg' in df.columns else 'longitude'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'
        
        df = df.sort_values([device_col, time_col]).reset_index(drop=True)
        
        # Mark original messages as normal
        df['Is_Attack'] = 0
        df['Attack_Label'] = 'Benign'
        
        # Get source vehicles
        if target_vehicles is None:
            target_vehicles = df[device_col].unique()
        
        # Calculate target replay message count
        target_replay_msgs = int(len(df) * attack_ratio)
        
        replayed_sequences = []
        total_replayed = 0
        ghost_vehicle_counter = 0
        
        # Conversion constants for position shift
        from bsm_attacker.geo_constants import DEG_LAT_TO_M, deg_lon_to_m
        meters_per_degree_lat = DEG_LAT_TO_M
        meters_per_degree_lon = deg_lon_to_m(df[lat_col].median())
        
        # Shuffle vehicles for random selection
        vehicle_list = list(target_vehicles)
        np.random.shuffle(vehicle_list)
        
        for source_vehicle in vehicle_list:
            if total_replayed >= target_replay_msgs:
                break
            
            vehicle_df = df[df[device_col] == source_vehicle].copy()
            
            if len(vehicle_df) < replay_sequence_length[0]:
                continue
            
            # Select random sequence from this vehicle
            seq_len = np.random.randint(
                replay_sequence_length[0],
                min(replay_sequence_length[1], len(vehicle_df)) + 1
            )
            
            max_start = len(vehicle_df) - seq_len
            if max_start <= 0:
                continue
                
            start_idx = np.random.randint(0, max_start)
            sequence = vehicle_df.iloc[start_idx:start_idx + seq_len].copy()
            
            # Apply replay modifications
            # 1. Time delay
            delay_ms = np.random.uniform(replay_delay_ms[0], replay_delay_ms[1])
            delay_s = delay_ms / 1000.0
            # Handle both datetime and numeric timestamp columns
            if pd.api.types.is_numeric_dtype(sequence[time_col]):
                # Numeric timestamps (epoch seconds): add delay directly
                sequence[time_col] = sequence[time_col].astype(float) + delay_s
            else:
                # Datetime/string timestamps: parse and add timedelta
                ts = pd.to_datetime(sequence[time_col], errors='coerce')
                sequence[time_col] = ts + pd.Timedelta(seconds=delay_s)
            
            # 2. New device ID (ghost vehicle)
            if modify_device_id:
                ghost_vehicle_counter += 1
                ghost_id = f"GHOST_{ghost_vehicle_counter:03d}"
                sequence[device_col] = ghost_id
            
            # 3. Position shift (parallel ghost)
            if position_shift_meters > 0:
                shift_direction = np.random.uniform(0, 2 * np.pi)
                delta_lat = (position_shift_meters * np.cos(shift_direction)) / meters_per_degree_lat
                delta_lon = (position_shift_meters * np.sin(shift_direction)) / meters_per_degree_lon
                sequence[lat_col] = sequence[lat_col] + delta_lat
                sequence[lon_col] = sequence[lon_col] + delta_lon
            
            # Mark as attack
            sequence['Is_Attack'] = 1
            sequence['Attack_Label'] = 'Data Replay'
            sequence['Replay_Source'] = source_vehicle
            sequence['Replay_Delay_ms'] = delay_ms
            
            replayed_sequences.append(sequence)
            total_replayed += len(sequence)
        
        # Combine original and replayed data
        if replayed_sequences:
            replay_df = pd.concat(replayed_sequences, ignore_index=True)
            df = pd.concat([df, replay_df], ignore_index=True)
            
            # Sort by timestamp — handle numeric and datetime types
            if pd.api.types.is_numeric_dtype(df[time_col]):
                df = df.sort_values(time_col).reset_index(drop=True)
            else:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.sort_values(time_col).reset_index(drop=True)
        
        # Fill NaN in replay-specific columns
        if 'Replay_Source' not in df.columns:
            df['Replay_Source'] = None
            df['Replay_Delay_ms'] = None
        else:
            df['Replay_Source'] = df['Replay_Source'].fillna('')
            df['Replay_Delay_ms'] = df['Replay_Delay_ms'].fillna(0)
        
        # Print summary
        attack_mask = df['Is_Attack'] == 1
        self.log_attack_summary(df, attack_mask)
        print(f"Replay sequences created: {len(replayed_sequences)}")
        print(f"Ghost vehicles created: {ghost_vehicle_counter}")
        print(f"Replay delay range: {replay_delay_ms[0]}-{replay_delay_ms[1]} ms")
        
        return df
