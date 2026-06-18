"""
Speed Limit Violation Attack
==========================================

Attacker broadcasts a speed significantly above the posted road speed limit
while *actually* driving at a legal or slower pace. Targets deployed Speed
Compliance (SpdComp) safety applications — the MOST commonly triggered app
in the NYC CV Pilot IE sample (5,951 events vs 2,650 FCW events).

Rationale:
  * SpdComp fires when ASD-reported speed exceeds posted limit by a
    configurable margin. An attacker broadcasting inflated speed would
    cause the victim's dashboard to show spurious "over-limit" warnings,
    or cause fleet management systems to flag false speeding incidents.
  * Conversely, an attacker broadcasting DEFLATED speed while speeding
    would evade SpdComp detection on their own vehicle (but we don't
    attack that direction here — we focus on injected "speeding" signals
    that create false positives for victims).

Scenario gating:
  * Requires `speed_limit_kmh` column (from OSM enrichment).
    Skip rows where the column is NaN (OSM has no maxspeed tag for that
    road segment). Coverage: 28-31% on US sites, 68-71% on EU sites.
  * Requires `Scenario_Label = 'Cruising'` — speeding while parked is
    implausible.
  * Requires real speed > 3 m/s so the attack signal is credible.

Co-spoofing:
  * `speed_mps`: multiply by `speed_multiplier` (default 1.5-2.0× posted limit)
  * `accel_long_mps2`: set to a small positive (0.0-0.5 m/s²) to imply
    sustained cruising at the higher speed rather than acceleration event.
  * Position: preserved — attacker's actual location doesn't change
  * Heading: preserved

Target safety app: SpdComp (Speed Compliance) — new coverage vs VeReMi/VASP.
"""

import numpy as np
import pandas as pd

from .base import BaseAttacker, get_column_name


class SpeedLimitViolationAttacker(BaseAttacker):
    """Implements the 'Speed Limit Violation' attack."""

    def __init__(self, random_seed: int = 42):
        super().__init__(attack_name="Speed Limit Violation",
                         random_seed=random_seed)

    def inject_attack(self, df: pd.DataFrame,
                      attack_ratio: float = 0.3,
                      speed_multiplier_range: tuple = (1.5, 2.0),
                      min_real_speed_mps: float = 3.0,
                      spoofed_accel_range: tuple = (0.0, 0.5),
                      target_vehicles: list | None = None) -> pd.DataFrame:
        """Inject Speed Limit Violation attacks.

        Args
        ----
        attack_ratio : fraction of eligible rows to attack.
        speed_multiplier_range : (min, max) multiple of posted limit. Spoofed
            speed = real_speed capped to `limit * multiplier`.
        min_real_speed_mps : require real speed above this to attack (avoid
            attacking near-stopped vehicles where a "speeding" signal is
            implausible).
        spoofed_accel_range : (min, max) m/s² to set as the co-spoofed accel.
        target_vehicles : restrict to given device IDs, else all.

        Requires columns: `speed_limit_kmh` (from enrichment) and
        `Scenario_Label`.
        """
        df = df.copy()
        rng = np.random.default_rng(self.random_seed)

        device_col = get_column_name(df, "device_id") or "Device_ID"
        time_col = get_column_name(df, "timestamp") or "Tx_Timestamp"
        speed_col = get_column_name(df, "speed") or "Speed_mps"
        accel_col = get_column_name(df, "accel_long") or "Accel_Long_mps2"
        scenario_col = get_column_name(df, "scenario_label") or "Scenario_Label"

        missing = []
        if "speed_limit_kmh" not in df.columns:
            missing.append("speed_limit_kmh")
        if scenario_col not in df.columns:
            missing.append(scenario_col)
        if missing:
            raise ValueError(
                f"Speed Limit Violation attack requires columns: {missing}. "
                f"Run OSM enrichment + scenario labeling first."
            )

        df = df.sort_values([device_col, time_col]).reset_index(drop=True)

        posted_limit_mps = df["speed_limit_kmh"] / 3.6  # km/h → m/s

        eligible = (
            df["speed_limit_kmh"].notna()
            & (df[scenario_col] == "Cruising")
            & (df[speed_col] >= min_real_speed_mps)
            & (df[speed_col] < posted_limit_mps * 1.2)  # not already speeding hard
        )

        if target_vehicles is None:
            target_vehicles = df[device_col].unique()

        attack_mask = pd.Series(False, index=df.index)
        orig_speed = df[speed_col].copy()
        orig_accel = df[accel_col].copy()

        for vid in target_vehicles:
            veh_mask = (df[device_col] == vid) & eligible
            eligible_idx = df.index[veh_mask].to_numpy()
            if len(eligible_idx) == 0:
                continue

            n_attack = max(1, int(len(eligible_idx) * attack_ratio))
            chosen = rng.choice(eligible_idx, size=n_attack, replace=False)

            multiplier = rng.uniform(
                speed_multiplier_range[0], speed_multiplier_range[1],
                size=n_attack,
            )
            spoof_speed = posted_limit_mps.loc[chosen].to_numpy() * multiplier
            # Clamp spoofed speed to a reasonable driving ceiling (60 m/s ≈ 216 km/h)
            spoof_speed = np.clip(spoof_speed, None, 60.0)

            spoof_accel = rng.uniform(
                spoofed_accel_range[0], spoofed_accel_range[1],
                size=n_attack,
            )

            df.loc[chosen, speed_col] = spoof_speed
            df.loc[chosen, accel_col] = spoof_accel
            attack_mask.loc[chosen] = True

        df = self.add_attack_labels(df, attack_mask, self.attack_name)
        df["Original_Speed_mps"] = orig_speed
        df["Original_Accel_Long"] = orig_accel

        self.log_attack_summary(df, attack_mask)
        n_eligible_total = int(eligible.sum())
        n_attacked = int(attack_mask.sum())
        print(f"  Eligible rows (Cruising + has maxspeed + moving): {n_eligible_total:,}")
        print(f"  Spoofed speed range: {speed_multiplier_range[0]}-"
              f"{speed_multiplier_range[1]}× posted limit")
        print(f"  Attack rows: {n_attacked:,}")
        return df
