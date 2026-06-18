"""
Intersection Bluff Attack (v2.0)
=================================

This attack targets vehicles in TURNING scenarios (e.g., executing turns at intersections).
The attacker aggressively spoofs their position to appear in the path of other vehicles,
potentially causing emergency braking or collision avoidance maneuvers.

Attack Mechanism:
- Target: Messages with Scenario_Label = 'Turning'
- Modification: Offset position perpendicular to heading by 5-15 meters (simulating lane intrusion)
- Realism: Consistent direction during turn + progressive offset buildup

v2.0 Co-spoofing:
- Tilts heading 2-5° toward offset direction, proportional to offset magnitude.
  Without this, heading stays on real turn arc while position drifts laterally,
  creating a heading/position inconsistency detectable by IDS.

Literature Reference:
- VeReMi "Random Offset" attack type, but scenario-aware (turning only)
- Key improvement: Consistent lateral direction per turn sequence (not random per message)
- Progressive offset buildup to mimic gradual lane departure

Use Case:
This attack exploits the natural position uncertainty during turning maneuvers,
making it harder to distinguish from GPS errors while creating dangerous situations.
"""

import pandas as pd
import numpy as np
from .base import BaseAttacker, get_column_name


class LateralDriftAtTurningAttacker(BaseAttacker):
    """
    Implements the 'Intersection Bluff' attack for turning scenarios.
    
    Improvement: Consistent lateral offset direction per turn sequence
    with progressive buildup, mimicking gradual lane departure.
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Lateral Drift At Turning", random_seed=random_seed)
    
    def inject_attack(self, df: pd.DataFrame,
                     attack_ratio: float = 0.5,
                     offset_range: tuple = (5, 15),
                     progressive_buildup: bool = True,
                     heading_tilt_factor: float = 0.4,
                     max_heading_tilt: float = 5.0,
                     target_vehicles: list = None) -> pd.DataFrame:
        """
        Inject 'Intersection Bluff' attacks into turning messages.

        v2.0: Co-spoofs heading toward offset direction to maintain
        heading/position consistency during lateral drift.

        Args:
            df: Input DataFrame with BSM data (must have 'Scenario_Label' column)
            attack_ratio: Proportion of turning messages to attack (0.0 to 1.0)
            offset_range: Tuple of (min_offset, max_offset) in meters for lateral position spoofing
            progressive_buildup: If True, offset builds up progressively during turn sequence
            heading_tilt_factor: Degrees of heading tilt per metre of offset (default 0.4)
            max_heading_tilt: Maximum heading tilt in degrees (default 5.0)
            target_vehicles: List of Device_IDs to attack (None = attack all vehicles)

        Returns:
            DataFrame with injected attacks
        """
        df = df.copy()

        device_col = get_column_name(df, 'device_id') or 'Device_ID'
        time_col = get_column_name(df, 'timestamp') or 'Tx_Timestamp'
        lat_col = get_column_name(df, 'latitude') or 'Latitude_deg'
        lon_col = get_column_name(df, 'longitude') or 'Longitude_deg'
        heading_col = get_column_name(df, 'heading') or 'Heading_deg'
        scenario_col = get_column_name(df, 'scenario_label') or 'Scenario_Label'

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        if scenario_col not in df.columns:
            raise ValueError(f"DataFrame must have '{scenario_col}' column. Run scenario labeling first.")

        attack_mask = pd.Series(False, index=df.index)

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        # lateral-lane intrusion requires (1) a real intersection
        # (not a curvy rural road), (2) a conflict lane to drift into — so
        # oneway roads and single-lane roads are excluded, (3) NOT a roundabout
        # (which has distinct curvature dynamics and no cross-conflict lanes).
        # In deployment all four bits come from J2735 MAP (polygon / laneSet /
        # lane connections / junction topology); we proxy offline via OSM
        # in_intersection_zone, lanes, oneway, junction=roundabout. VeReMi
        # synthetic lacks these → fallback to kinematic-only Turning.
        has_osm = all(c in df.columns for c in
                      ('in_intersection_zone', 'lanes', 'oneway', 'is_roundabout'))
        if has_osm:
            context_ok = (
                df['in_intersection_zone'].fillna(False).astype(bool)
                & (~df['oneway'].fillna(False).astype(bool))
                & (df['lanes'].fillna(0).astype(float) >= 2)
                & (~df['is_roundabout'].fillna(False).astype(bool))
            )
        elif 'in_intersection_zone' in df.columns:
            context_ok = df['in_intersection_zone'].fillna(False).astype(bool)
        else:
            context_ok = pd.Series(True, index=df.index)

        total_turning = 0
        total_attacked = 0

        for vehicle_id in target_vehicles:
            vehicle_mask = (df[device_col] == vehicle_id) & (df[scenario_col] == 'Turning') & context_ok
            vehicle_turning_indices = df[vehicle_mask].index.tolist()

            if len(vehicle_turning_indices) == 0:
                continue

            total_turning += len(vehicle_turning_indices)
            segments = self._find_contiguous_segments(vehicle_turning_indices)

            for segment in segments:
                num_to_attack = max(1, int(len(segment) * attack_ratio))
                attack_indices = segment[:num_to_attack]

                if not attack_indices:
                    continue

                total_attacked += len(attack_indices)
                direction = np.random.choice([-1, 1])
                max_offset = np.random.uniform(offset_range[0], offset_range[1])

                for i, idx in enumerate(attack_indices):
                    lat = df.loc[idx, lat_col]
                    lon = df.loc[idx, lon_col]
                    heading = df.loc[idx, heading_col]

                    if progressive_buildup:
                        progress = (i + 1) / len(attack_indices)
                        offset_distance = max_offset * progress
                    else:
                        offset_distance = max_offset

                    # Position: perpendicular offset
                    perpendicular_heading = heading + (90 * direction)
                    offset_north = offset_distance * np.cos(np.radians(perpendicular_heading))
                    offset_east = offset_distance * np.sin(np.radians(perpendicular_heading))
                    new_lat, new_lon = self.offset_coordinates(lat, lon, offset_north, offset_east)
                    df.loc[idx, lat_col] = new_lat
                    df.loc[idx, lon_col] = new_lon

                    # v2.0: Heading tilt toward offset direction
                    heading_tilt = direction * min(max_heading_tilt, offset_distance * heading_tilt_factor)
                    df.loc[idx, heading_col] = heading + heading_tilt

                    attack_mask.loc[idx] = True

        if total_turning == 0:
            print(f"Warning: No turning messages found to attack. Returning original DataFrame.")
            return self.add_attack_labels(df, attack_mask, self.attack_name)

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        self.log_attack_summary(df, attack_mask)

        return df
    
    def _find_contiguous_segments(self, indices: list) -> list:
        """
        Find contiguous segments in a list of indices.
        
        Returns list of lists, where each inner list is a contiguous segment.
        """
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
