"""
False Yield at Intersection Attack
================================================

Context-aware IMA attack. Attacker fakes a yielding / near-stop behavior
when approaching an intersection (broadcasting low speed + negative accel)
while *actually* continuing at normal speed. Victim vehicles whose IMA
consumes the attacker's BSMs see a vehicle that appears to be stopping,
so IMA suppresses its warning. The victim proceeds. The attacker — still
moving at real speed — arrives at the intersection, causing a potential
broadside collision.

Maps to VASP "Low Speed IMA" (Ansari 2023 §IV.C, HIGH RISK):
    "Attacker transmits lower speed values to throw off TTC and DTC
     calculations. This attack aims at making other vehicles think that
     the attacker is approaching the intersection later than in reality.
     This could cause fatal accidents at the intersection."

Differs from existing context-aware attacks:
  * `liar_at_light` — offsets position forward *after* stopping
  * `heading_lock` — fakes straight-through during an actual turn
  * THIS — fakes yielding during an actual straight approach

Scenario gating:
  1. Preferred: `dist_to_intersection_m <= 100` AND `heading_diff_deg <= 45`
     (approaching an intersection with aligned heading). Requires
     Stage 1 annotated columns.
  2. Fallback: `Scenario_Label == 'Cruising'` followed by Stationary_Wait
     on the same device — detects "vehicle slowed before a stop" sequences.

Co-spoofing (multi-field consistency):
  * `speed_mps`: ramp from real value → 1.0 m/s over spoof window
  * `accel_long_mps2`: sustained -2.0 to -3.0 m/s² (consistent with speed ramp)
  * Position: preserved — attacker is actually moving
  * Heading: preserved

Target safety app: IMA
"""

import numpy as np
import pandas as pd

from .base import BaseAttacker, get_column_name


class FalseYieldAtIntersectionAttacker(BaseAttacker):
    """Implements the 'False Yield at Intersection' attack."""

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="False Yield At Intersection",
                         random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.5,
                      ramp_msgs: int = 6,
                      target_speed_mps: float = 1.0,
                      decel_range: tuple = (-3.0, -2.0),
                      max_dist_to_intersection_m: float = 100.0,
                      max_heading_diff_deg: float = 45.0,
                      target_vehicles: list | None = None) -> pd.DataFrame:
        """Inject False-Yield-at-Intersection attacks.

        Args
        ----
        attack_ratio : fraction of eligible intersection-approach sequences to attack.
        ramp_msgs : number of BSMs over which speed ramps from real → target.
        target_speed_mps : terminal spoofed speed during the "yield" (near-zero).
        decel_range : (min, max) m/s² for sustained deceleration during spoof.
        max_dist_to_intersection_m : scenario gate (needs annotation).
        max_heading_diff_deg : angular alignment gate for "approaching" heuristic.
        target_vehicles : restrict to given device IDs, else all.
        """
        df = df.copy()
        rng = np.random.default_rng(self.random_seed)

        device_col = get_column_name(df, "device_id") or "Device_ID"
        time_col = get_column_name(df, "timestamp") or "Tx_Timestamp"
        speed_col = get_column_name(df, "speed") or "Speed_mps"
        accel_col = get_column_name(df, "accel_long") or "Accel_Long_mps2"

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        # Eligibility mask: near intersection + approaching + moving.
        # refinement — IMA's scope per VSC-A DOT HS 811 466 and
        # Iteris CVRIA App 36 is stop-controlled and UNCONTROLLED intersections
        # (signalized intersections get SLTA, not IMA). Exclude roundabouts
        # (different negotiation pattern). Deployment mapping: is_stop_controlled
        # from MAP-absent+OSM stop-sign tag, is_signalized from MAP signal group.
        if ("dist_to_intersection_m" in df.columns
                and "heading_diff_deg" in df.columns):
            base = (
                df["dist_to_intersection_m"].notna()
                & (df["dist_to_intersection_m"] <= max_dist_to_intersection_m)
                & df["heading_diff_deg"].notna()
                & (df["heading_diff_deg"] <= max_heading_diff_deg)
                & (df[speed_col] >= 2.0)
            )
            # Scope: stop-controlled OR plain-intersection (uncontrolled),
            # never signalized, never roundabout. If is_signalized /
            # is_stop_controlled / is_roundabout aren't present, skip the
            # tightening (fall back to base).
            has_typing = all(c in df.columns for c in
                             ('is_signalized', 'is_stop_controlled', 'is_roundabout'))
            if has_typing:
                ima_scope = (
                    df['is_stop_controlled'].fillna(False).astype(bool)
                    | (df['in_intersection_zone'].fillna(False).astype(bool)
                       & ~df['is_signalized'].fillna(False).astype(bool))
                ) & ~df['is_roundabout'].fillna(False).astype(bool)
                eligible = base & ima_scope
            else:
                eligible = base
        else:
            # Fallback: use Scenario_Label = 'Cruising'. Coarser but functional.
            scenario_col = get_column_name(df, "scenario_label") or "Scenario_Label"
            if scenario_col not in df.columns:
                raise ValueError(
                    "False-Yield-at-Intersection requires either "
                    "annotation columns (dist_to_intersection_m, heading_diff_deg) "
                    "or Scenario_Label for fallback gating."
                )
            eligible = (
                (df[scenario_col] == "Cruising") & (df[speed_col] >= 2.0)
            )

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        attack_mask = pd.Series(False, index=df.index)
        orig_speed = df[speed_col].copy()
        orig_accel = df[accel_col].copy()

        for vid in target_vehicles:
            veh_mask = (df[device_col] == vid) & eligible
            eligible_idx = df.index[veh_mask].to_list()
            if len(eligible_idx) < ramp_msgs:
                continue

            # Pick random contiguous window(s) from eligible rows.
            # For simplicity, attack one window per eligible run of >= ramp_msgs.
            # Split eligible indices into contiguous runs
            runs = []
            if eligible_idx:
                cur = [eligible_idx[0]]
                for i in eligible_idx[1:]:
                    if i == cur[-1] + 1:
                        cur.append(i)
                    else:
                        if len(cur) >= ramp_msgs:
                            runs.append(cur)
                        cur = [i]
                if len(cur) >= ramp_msgs:
                    runs.append(cur)

            n_attack = max(1, int(len(runs) * attack_ratio))
            if n_attack < len(runs):
                chosen = rng.choice(len(runs), size=n_attack, replace=False)
                runs = [runs[i] for i in chosen]

            for run in runs:
                # Use first `ramp_msgs` of the run for the yield ramp
                win = run[:ramp_msgs]
                start_v = float(df.loc[win[0], speed_col])
                # Linear ramp from start_v to target_speed_mps
                ramp = np.linspace(start_v, target_speed_mps, len(win))
                decel = rng.uniform(decel_range[0], decel_range[1])
                df.loc[win, speed_col] = ramp
                df.loc[win, accel_col] = decel
                attack_mask.loc[win] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df["Original_Speed_mps"] = orig_speed
        df["Original_Accel_Long"] = orig_accel

        self.log_attack_summary(df, attack_mask)
        print(f"  Yield ramp: {ramp_msgs} msgs to {target_speed_mps:.1f} m/s "
              f"with decel {decel_range[0]} to {decel_range[1]} m/s²")
        return df
