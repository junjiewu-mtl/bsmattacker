"""
Eventual Stop Attack (VeReMi Type 16)
=====================================

This attack gradually slows down the reported speed until the vehicle appears to stop,
while the actual vehicle continues moving at normal speed.

Attack Mechanism:
- Target: Vehicles in CRUISING scenarios (moving vehicles)
- Modification: Gradual speed reduction to zero over time
- Goal: Cause following vehicles to brake/change lanes for a "stopped" vehicle that isn't there

Literature Reference:
- VeReMi Dataset: Type 16 "Eventual Stop" attack
- The attacker falsely reports decreasing speed until 0 m/s
- Creates phantom stopped vehicle hazard on roadway

Use Case:
This attack exploits trust in BSM for collision avoidance systems.
A following vehicle with FCW/AEB may brake hard for a "stopped" vehicle that doesn't exist.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class EventualStopAttacker(BaseAttacker):
    """
    Implements the 'Eventual Stop' attack (VeReMi Type 16).
    
    The attacker reports gradually decreasing speed until the vehicle
    appears stopped, while it continues moving normally.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Eventual Stop", random_seed=random_seed)
    
    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.15,
                     stop_duration_msgs: tuple = (5, 15),
                     decel_rate: tuple = (-4.0, -2.0),
                     hold_stopped_msgs: int = 5,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Eventual Stop' attacks into BSM data.
        
        Attack Profile:
        1. Select random cruising segments
        2. Gradually reduce speed to 0 with consistent deceleration
        3. Hold at 0 speed for several messages (phantom stopped vehicle)
        
        Args:
            df: Input DataFrame with BSM data
            attack_ratio: Proportion of cruising messages to attack (0.0 to 1.0)
            stop_duration_msgs: Tuple (min, max) messages for speed reduction phase
            decel_rate: Tuple (min, max) deceleration in m/s² 
                        Default (-4.0, -2.0) for realistic braking
            hold_stopped_msgs: Number of messages to hold at speed=0 after stopping
            target_vehicles: List of Device_IDs to attack (None = attack all vehicles)
            
        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()
        
        # Determine column names (support both 22-col and 26-col formats)
        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        speed_col = 'Speed_mps' if 'Speed_mps' in df.columns else 'speed_mps'
        accel_col = 'Accel_Long_mps2' if 'Accel_Long_mps2' in df.columns else 'accel_long_ms2'
        time_col = 'Tx_Timestamp' if 'Tx_Timestamp' in df.columns else 'timestamp'
        
        df = df.sort_values([device_col, time_col]).reset_index(drop=True)
        
        # Add Scenario_Label if missing
        if 'Scenario_Label' not in df.columns:
            df = self._add_scenario_labels(df, speed_col)
        
        # Create attack columns
        attack_mask = pd.Series(False, index=df.index)
        original_speeds = df[speed_col].copy()
        original_accels = df[accel_col].copy() if accel_col in df.columns else None
        
        # Get target vehicles
        if target_vehicles is None:
            target_vehicles = df[device_col].unique()
        
        total_cruising = 0
        total_attacked = 0
        attack_sequences = 0
        contributing_vehicles = 0

        # Process each vehicle separately
        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (df['Scenario_Label'] == 'Cruising')
            vehicle_cruising_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_cruising_indices) == 0:
                continue

            total_cruising += len(vehicle_cruising_indices)

            # Find contiguous cruising segments
            segments = self._find_contiguous_segments(vehicle_cruising_indices)

            # Filter segments long enough for eventual stop attack
            min_segment_len = stop_duration_msgs[1] + hold_stopped_msgs
            valid_segments = [s for s in segments if len(s) >= min_segment_len]

            if not valid_segments:
                continue

            # per-vehicle attack budget (was global accumulator bug
            # where `total_attacked >= target_attack_msgs` immediately tripped
            # for every vehicle after the first; only 15/5179 attackers contributed).
            target_attack_msgs = int(len(vehicle_cruising_indices) * attack_ratio)
            vehicle_attacked = 0

            # Shuffle and select segments
            np.random.shuffle(valid_segments)

            for segment in valid_segments:
                if vehicle_attacked >= target_attack_msgs:
                    break
                
                # Random stop duration within range
                stop_duration = np.random.randint(stop_duration_msgs[0], stop_duration_msgs[1] + 1)
                total_attack_len = stop_duration + hold_stopped_msgs
                
                if len(segment) < total_attack_len:
                    continue
                
                # Start position within segment
                max_start = len(segment) - total_attack_len
                start_pos = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                attack_indices = segment[start_pos:start_pos + total_attack_len]
                
                # Get initial speed
                initial_speed = df.loc[attack_indices[0], speed_col]
                if pd.isna(initial_speed) or initial_speed < 1.0:
                    continue  # Skip if already slow
                
                # Random deceleration rate
                decel = np.random.uniform(decel_rate[0], decel_rate[1])
                
                # Phase 1: Gradual speed reduction
                current_speed = initial_speed
                for i, idx in enumerate(attack_indices[:stop_duration]):
                    # Calculate speed reduction
                    # Assuming ~100ms between messages (10Hz BSM rate)
                    dt = 0.1
                    current_speed = max(0, current_speed + decel * dt)
                    
                    df.loc[idx, speed_col] = current_speed
                    if accel_col in df.columns:
                        df.loc[idx, accel_col] = decel if current_speed > 0 else 0
                    
                    attack_mask.loc[idx] = True
                
                # Phase 2: Hold at stopped
                for idx in attack_indices[stop_duration:]:
                    df.loc[idx, speed_col] = 0.0
                    if accel_col in df.columns:
                        df.loc[idx, accel_col] = 0.0
                    attack_mask.loc[idx] = True
                
                total_attacked += len(attack_indices)
                vehicle_attacked += len(attack_indices)
                attack_sequences += 1

            if vehicle_attacked > 0:
                contributing_vehicles += 1

        # Add attack labels
        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df['Original_Speed'] = original_speeds
        if original_accels is not None:
            df['Original_Accel'] = original_accels

        # Print summary
        self.log_attack_summary(df, attack_mask)
        print(f"Attack sequences injected: {attack_sequences}")
        print(f"Contributing attacker vehicles: {contributing_vehicles} / "
              f"{len(target_vehicles)} ( per-vehicle counter)")

        return df
    
    def _find_contiguous_segments(self, indices: list) -> list:
        """Find contiguous segments in a list of indices."""
        if not indices:
            return []
        
        segments = []
        current_segment = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]
        
        segments.append(current_segment)
        return segments
    
    def _add_scenario_labels(self, df: pd.DataFrame, speed_col: str) -> pd.DataFrame:
        """Add Scenario_Label column based on speed."""
        df['Scenario_Label'] = 'Cruising'
        df.loc[df[speed_col] < 0.5, 'Scenario_Label'] = 'Stationary'
        
        # Check for turning via yaw rate if available
        yaw_col = 'Yaw_Rate' if 'Yaw_Rate' in df.columns else 'yaw_rate_degs'
        if yaw_col in df.columns:
            df.loc[df[yaw_col].abs() > 5.0, 'Scenario_Label'] = 'Turning'
        
        return df
