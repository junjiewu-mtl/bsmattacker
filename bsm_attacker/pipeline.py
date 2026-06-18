"""
Attack Pipeline
===============

Provides a high-level interface for injecting multiple attacks into BSM data
with configurable parameters and automatic result tracking.

Supports both legacy and extended CSV column formats.

Attack Types:
- Context-aware: position_pullthrough_at_stop, lateral_drift_at_turning,
  false_deceleration_cruising, slow_position_drift,
  phantom_acceleration, heading_lock
- VeReMi-Equivalent: constant_position, constant_position_offset, random_position,
  random_position_offset, constant_speed, random_speed, eventual_stop, data_replay
- VASP Speed Offsets: constant_speed_offset, random_speed_offset
- VASP Heading: constant_heading, constant_heading_offset, random_heading,
  random_heading_offset, opposite_heading, perpendicular_heading
- VASP Acceleration: constant_acceleration, random_acceleration, random_acceleration_offset
- VASP Position: ghost_vehicle
- Safety-app-aligned additions: false_yield_at_intersection, speed_limit_violation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json

# Original attacks (IEEE TITS-style names — rename)
from .position_pullthrough_at_stop import PositionPullthroughAtStopAttacker
from .lateral_drift_at_turning import LateralDriftAtTurningAttacker
from .false_deceleration_cruising import FalseDecelerationCruisingAttacker
from .slow_position_drift import SlowPositionDriftAttacker
from .phantom_acceleration import PhantomAccelerationAttacker
from .heading_lock import HeadingLockAttacker

# VeReMi-equivalent attacks
from .constant_position import ConstantPositionAttacker
from .constant_position_offset import ConstantPositionOffsetAttacker
from .random_position import RandomPositionAttacker
from .random_position_offset import RandomPositionOffsetAttacker
from .constant_speed import ConstantSpeedAttacker
from .random_speed import RandomSpeedAttacker
from .eventual_stop import EventualStopAttacker
from .data_replay import DataReplayAttacker

# VASP-inspired attacks (speed offsets)
from .constant_speed_offset import ConstantSpeedOffsetAttacker
from .random_speed_offset import RandomSpeedOffsetAttacker

# VASP-inspired attacks (heading)
from .constant_heading import ConstantHeadingAttacker
from .constant_heading_offset import ConstantHeadingOffsetAttacker
from .random_heading import RandomHeadingAttacker
from .random_heading_offset import RandomHeadingOffsetAttacker
from .opposite_heading import OppositeHeadingAttacker
from .perpendicular_heading import PerpendicularHeadingAttacker

# VASP-inspired attacks (acceleration)
from .constant_acceleration import ConstantAccelerationAttacker
from .random_acceleration import RandomAccelerationAttacker
from .random_acceleration_offset import RandomAccelerationOffsetAttacker

# VASP-inspired attacks (position)
from .ghost_vehicle import GhostVehicleAttacker

# safety-app-aligned additions
from .false_yield_at_intersection import FalseYieldAtIntersectionAttacker
from .speed_limit_violation import SpeedLimitViolationAttacker

from .base import detect_csv_format, get_column_name, COLUMN_ALIASES

_GLOBAL_SEMANTICS_WARNING_EMITTED = False

CANONICAL_ATTACK_TYPES = (
    # Context-aware attacks, using the current paper-facing names.
    "position_pullthrough_at_stop",
    "lateral_drift_at_turning",
    "false_deceleration_cruising",
    "slow_position_drift",
    "phantom_acceleration",
    "heading_lock",
    # VeReMi-equivalent attacks.
    "constant_position",
    "constant_position_offset",
    "random_position",
    "random_position_offset",
    "constant_speed",
    "random_speed",
    "eventual_stop",
    "data_replay",
    # VASP-inspired attacks.
    "constant_speed_offset",
    "random_speed_offset",
    "constant_heading",
    "constant_heading_offset",
    "random_heading",
    "random_heading_offset",
    "opposite_heading",
    "perpendicular_heading",
    "constant_acceleration",
    "random_acceleration",
    "random_acceleration_offset",
    "ghost_vehicle",
    # Enrichment-dependent safety-app additions.
    "false_yield_at_intersection",
    "speed_limit_violation",
)

class AttackPipeline:
    """
    High-level pipeline for injecting multiple attacks into BSM data.
    Supports both legacy and extended CSV column formats.

    Attack types available: 28 canonical names.
    - Context-aware: position_pullthrough_at_stop, lateral_drift_at_turning,
      false_deceleration_cruising, slow_position_drift,
      phantom_acceleration, heading_lock
    - VeReMi-equivalent: constant_position, constant_position_offset, random_position,
      random_position_offset, constant_speed, random_speed, eventual_stop, data_replay
    - VASP speed offsets: constant_speed_offset, random_speed_offset
    - VASP heading: constant_heading, constant_heading_offset, random_heading,
      random_heading_offset, opposite_heading, perpendicular_heading
    - VASP acceleration: constant_acceleration, random_acceleration, random_acceleration_offset
    - VASP position: ghost_vehicle
    - Safety-app additions: false_yield_at_intersection, speed_limit_violation
    """
    
    def __init__(self, random_seed: int = 42, show_semantics_warning: bool = False):
        """
        Initialize the attack pipeline.

        Args:
            random_seed: Random seed for reproducibility
            show_semantics_warning: Emit a one-time note that direct class-based
                injection applies attacker logic per row, which differs from
                segment-assignment benchmarking.
        """
        self.random_seed = random_seed
        self.show_semantics_warning = show_semantics_warning
        self._semantics_warning_emitted = False
        self.canonical_attack_types = list(CANONICAL_ATTACK_TYPES)

        # Original context-aware attacks (6)
        self.attackers = {
            'position_pullthrough_at_stop': PositionPullthroughAtStopAttacker(random_seed=random_seed),
            'lateral_drift_at_turning': LateralDriftAtTurningAttacker(random_seed=random_seed),
            'false_deceleration_cruising': FalseDecelerationCruisingAttacker(random_seed=random_seed),
            'slow_position_drift': SlowPositionDriftAttacker(random_seed=random_seed),
            'phantom_acceleration': PhantomAccelerationAttacker(random_seed=random_seed),
            'heading_lock': HeadingLockAttacker(random_seed=random_seed),
            # VeReMi-equivalent attacks
            'constant_position': ConstantPositionAttacker(random_seed=random_seed),
            'constant_position_offset': ConstantPositionOffsetAttacker(random_seed=random_seed),
            'random_position': RandomPositionAttacker(random_seed=random_seed),
            'random_position_offset': RandomPositionOffsetAttacker(random_seed=random_seed),
            'constant_speed': ConstantSpeedAttacker(random_seed=random_seed),
            'random_speed': RandomSpeedAttacker(random_seed=random_seed),
            'eventual_stop': EventualStopAttacker(random_seed=random_seed),
            'data_replay': DataReplayAttacker(random_seed=random_seed),
            # VASP-inspired attacks (speed offsets)
            'constant_speed_offset': ConstantSpeedOffsetAttacker(random_seed=random_seed),
            'random_speed_offset': RandomSpeedOffsetAttacker(random_seed=random_seed),
            # VASP-inspired attacks (heading)
            'constant_heading': ConstantHeadingAttacker(random_seed=random_seed),
            'constant_heading_offset': ConstantHeadingOffsetAttacker(random_seed=random_seed),
            'random_heading': RandomHeadingAttacker(random_seed=random_seed),
            'random_heading_offset': RandomHeadingOffsetAttacker(random_seed=random_seed),
            'opposite_heading': OppositeHeadingAttacker(random_seed=random_seed),
            'perpendicular_heading': PerpendicularHeadingAttacker(random_seed=random_seed),
            # VASP-inspired attacks (acceleration)
            'constant_acceleration': ConstantAccelerationAttacker(random_seed=random_seed),
            'random_acceleration': RandomAccelerationAttacker(random_seed=random_seed),
            'random_acceleration_offset': RandomAccelerationOffsetAttacker(random_seed=random_seed),
            # VASP-inspired attacks (position)
            'ghost_vehicle': GhostVehicleAttacker(random_seed=random_seed),
            # safety-app-aligned additions
            'false_yield_at_intersection': FalseYieldAtIntersectionAttacker(random_seed=random_seed),
            'speed_limit_violation': SpeedLimitViolationAttacker(random_seed=random_seed),
        }
        self.attack_history = []
        self.csv_format = None

    def _emit_semantics_warning(self):
        """Warn once that class-path injection differs from segment-assignment benchmarking."""
        global _GLOBAL_SEMANTICS_WARNING_EMITTED
        if not self.show_semantics_warning or self._semantics_warning_emitted:
            return
        if _GLOBAL_SEMANTICS_WARNING_EMITTED:
            self._semantics_warning_emitted = True
            return
        print(
            "Note: AttackPipeline.inject_single_attack() applies attacker-class logic "
            "directly per row and is not equivalent to segment-assignment benchmark "
            "injection, which uses benign_fraction/segment_size. Use the benchmark "
            "injection path for paper-equivalent results."
        )
        self._semantics_warning_emitted = True
        _GLOBAL_SEMANTICS_WARNING_EMITTED = True
    
    def preprocess_data(self, df: pd.DataFrame, add_scenario_labels: bool = True) -> pd.DataFrame:
        """
        Preprocess data to ensure compatibility with attack injection.
        Handles both 22-column and 26-column formats.
        
        Args:
            df: Input DataFrame with BSM data
            add_scenario_labels: If True, add Scenario_Label column if missing
            
        Returns:
            Preprocessed DataFrame ready for attack injection
        """
        df = df.copy()
        
        # Detect format
        self.csv_format = detect_csv_format(df)
        print(f"Detected CSV format: {self.csv_format}")
        
        # Normalize column names using COLUMN_ALIASES from base.py
        # Maps any known variant to the CamelCase target (first entry in each alias list)
        col_map = {}
        for _canonical, aliases in COLUMN_ALIASES.items():
            target = aliases[0]  # CamelCase target (e.g. Device_ID)
            if target in df.columns:
                continue  # Already has the target name
            for variant in aliases[1:]:
                if variant in df.columns:
                    col_map[variant] = target
                    break
        
        if col_map:
            df = df.rename(columns=col_map)
            print(f"Renamed columns: {col_map}")
        
        # Add Scenario_Label if missing (required for scenario-aware attacks)
        if add_scenario_labels and 'Scenario_Label' not in df.columns:
            df = self._add_scenario_labels(df)
        
        return df
    
    def _add_scenario_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Scenario_Label column with duration-aware Stationary sub-labels.

        Stationary messages are further split by contiguous-segment
        duration per device:
          - Stationary_Brief  (< 5 s):   stop signs, momentary pauses
          - Stationary_Wait   (5–60 s):  traffic lights, intersection waits
          - Stationary_Parked (> 60 s):  parking, pre-drive warmup

        Turning (|yaw_rate| > 5 deg/s) and Cruising labels are unchanged.
        """
        speed_col = next((c for c in ['Speed_mps', 'speed_mps'] if c in df.columns), 'speed_mps')
        yaw_col = next((c for c in ['Yaw_Rate', 'yaw_rate_degs', 'yaw_rate_dps'] if c in df.columns), 'yaw_rate_degs')
        device_col = next((c for c in ['Device_ID', 'device_id'] if c in df.columns), 'device_id')
        time_col = next((c for c in ['Tx_Timestamp', 'timestamp', 'timestamp_s'] if c in df.columns), 'timestamp')

        # --- Step 1: base labels ----------------------------------------
        df['Scenario_Label'] = 'Cruising'

        if speed_col in df.columns:
            df.loc[df[speed_col] < 0.5, 'Scenario_Label'] = 'Stationary'

        if yaw_col in df.columns:
            df.loc[df[yaw_col].abs() > 5.0, 'Scenario_Label'] = 'Turning'

        # --- Step 2: duration-aware Stationary sub-labels ---------------
        stationary_mask = df['Scenario_Label'] == 'Stationary'
        if stationary_mask.any() and device_col in df.columns:
            df['_is_stat'] = stationary_mask.astype(int)
            df['_stat_grp'] = (
                df.groupby(device_col)['_is_stat']
                .transform(lambda x: (x != x.shift()).cumsum())
            )

            # perf fix: vectorize the per-segment duration loop.
            # Prior per-segment Python for-loop with df.loc[seg.index,…] was
            # O(segments × n) on 1.9M rows / 17K devices — ~46 min per call.
            # Vectorized version: one groupby.agg + one row-level np.select.
            stat_only = df.loc[stationary_mask, [device_col, '_stat_grp']].copy()
            # Per-segment size + duration (aggregated once)
            if time_col in df.columns and pd.api.types.is_numeric_dtype(df[time_col]):
                stat_only[time_col] = df.loc[stationary_mask, time_col].values
                seg_stats = (
                    stat_only.groupby([device_col, '_stat_grp'])[time_col]
                    .agg(['min', 'max', 'count'])
                )
                seg_stats['duration_s'] = seg_stats['max'] - seg_stats['min']
            else:
                seg_stats = (
                    stat_only.groupby([device_col, '_stat_grp'])
                    .size()
                    .to_frame('count')
                )
                seg_stats['duration_s'] = (seg_stats['count'] - 1) * 0.1  # fallback 10 Hz

            # Assign sub-labels per-segment via vectorized np.select
            dur = seg_stats['duration_s'].values
            cnt = seg_stats['count'].values
            seg_labels = np.where(
                cnt < 2, 'Stationary_Brief',
                np.where(
                    dur < 5.0, 'Stationary_Brief',
                    np.where(dur <= 60.0, 'Stationary_Wait', 'Stationary_Parked'),
                ),
            )
            seg_stats['_label'] = seg_labels

            # Broadcast segment label to each row via merge (O(n log n))
            key_cols = df.loc[stationary_mask, [device_col, '_stat_grp']]
            row_labels = key_cols.merge(
                seg_stats['_label'].reset_index(),
                on=[device_col, '_stat_grp'],
                how='left',
            )['_label'].values
            df.loc[stationary_mask, 'Scenario_Label'] = row_labels

            df.drop(columns=['_is_stat', '_stat_grp'], inplace=True)

        scenario_counts = df['Scenario_Label'].value_counts()
        print(f"Scenario labels added: {scenario_counts.to_dict()}")

        return df
    
    def inject_single_attack(self, df: pd.DataFrame, attack_type: str, 
                            auto_preprocess: bool = True, **kwargs) -> pd.DataFrame:
        """
        Inject a single attack type into the dataset.
        
        Args:
            df: Input DataFrame with BSM data
            attack_type: Name of a registered attack type (see self.attackers
                for the available canonical attack names)
            auto_preprocess: If True, automatically preprocess data for compatibility
            **kwargs: Attack-specific parameters
            
        Returns:
            DataFrame with injected attack
        """
        if attack_type not in self.attackers:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                           f"Available: {list(self.attackers.keys())}")

        self._emit_semantics_warning()
        
        # Auto-preprocess to handle both 22-col and 26-col formats
        if auto_preprocess:
            df = self.preprocess_data(df)
        
        attacker = self.attackers[attack_type]
        df_attacked = attacker.inject_attack(df, **kwargs)
        
        # Record attack in history
        self.attack_history.append({
            'attack_type': attack_type,
            'parameters': kwargs,
            'total_messages': int(len(df_attacked)),
            'attacked_messages': int(df_attacked['Is_Attack'].sum()),
            'csv_format': self.csv_format
        })
        
        return df_attacked
    
    def inject_all_attacks(self, df: pd.DataFrame, 
                          attack_configs: Optional[Dict] = None,
                          auto_preprocess: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Inject all attack types into separate copies of the dataset.
        
        Args:
            df: Input DataFrame with BSM data (22-col or 26-col format)
            attack_configs: Dictionary of attack-specific configurations
                           Format: {attack_type: {param1: value1, ...}}
            auto_preprocess: If True, preprocess data once before all attacks
                           
        Returns:
            Dictionary mapping attack types to attacked DataFrames
        """
        if attack_configs is None:
            # Default: canonical attacks only. Deprecated aliases remain
            # dispatchable, but are not duplicated in "all attacks" sweeps.
            attack_configs = {
                name: {'attack_ratio': 0.2}
                for name in self.canonical_attack_types
            }

        self._emit_semantics_warning()
        
        # Preprocess data once for all attacks
        if auto_preprocess:
            df = self.preprocess_data(df)
            print(f"\nData preprocessed. Format: {self.csv_format}")
        
        results = {}
        
        for attack_type, config in attack_configs.items():
            print(f"\n{'='*70}")
            print(f"Injecting: {attack_type}")
            print(f"{'='*70}")
            
            df_copy = df.copy()
            # Skip preprocessing since we already did it
            results[attack_type] = self.inject_single_attack(df_copy, attack_type, 
                                                            auto_preprocess=False, **config)
        
        return results
    
    def save_attacked_datasets(self, results: Dict[str, pd.DataFrame], 
                              output_dir: str):
        """
        Save attacked datasets to CSV files.
        
        Args:
            results: Dictionary of attacked DataFrames from inject_all_attacks()
            output_dir: Directory to save the CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for attack_type, df in results.items():
            output_file = output_path / f"attacked_{attack_type}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
        
        # Save attack history as JSON
        history_file = output_path / "attack_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.attack_history, f, indent=2)
        print(f"Saved attack history: {history_file}")
    
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate a summary report of all attacks injected.
        
        Returns:
            DataFrame with attack statistics
        """
        if not self.attack_history:
            print("No attacks have been injected yet.")
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.attack_history)
        summary['attack_percentage'] = (summary['attacked_messages'] / 
                                        summary['total_messages'] * 100)
        
        return summary
