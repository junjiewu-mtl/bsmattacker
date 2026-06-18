"""
Realism Gate —
=========================
Filter devices whose benign trajectories are too short or lack the
scenario segments required for meaningful attack evaluation.

Thresholds derived from:
- Hard floor: seq_len=10, train_ratio=0.8 → minimum 50 rows per device
  to produce valid sliding windows in both train and test splits.
- Context-aware attacks: contiguous scenario segments for attack injection.
- eventual_stop: 20 consecutive Cruising for realistic deceleration.
"""

import numpy as np
import pandas as pd

# Minimum rows per device for valid train/test split + windowing
HARD_FLOOR = 50

# Scenario-gated attacks: (required_scenario_label, min_contiguous_length).
# Includes the 6 context-aware attacks plus eventual_stop (VeReMi-family but
# scope-gated to Cruising by the attacker class — see bsm_attacker/eventual_stop.py).
SCENARIO_GATED_REQUIREMENTS = {
    # IEEE TITS-style names ( rename — canonical going forward)
    "position_pullthrough_at_stop": ("Stationary_Wait", 5),
    "false_deceleration_cruising":  ("Cruising", 3),
    "slow_position_drift":          ("Cruising", 3),
    "phantom_acceleration":         ("Stationary_Brief", 5),
    "lateral_drift_at_turning":     ("Turning", 3),
    "heading_lock":                 ("Turning", 3),
    "eventual_stop":                ("Cruising", 20),
    # additions
    "false_yield_at_intersection":  ("Cruising", 3),
    # Back-compat aliases (deprecated)
    "liar_at_light":        ("Stationary_Wait", 5),
    "brake_bluff":          ("Cruising", 3),
    "stealth_drift":        ("Cruising", 3),
    "intersection_bluff":   ("Turning", 3),
}

# Backwards-compatibility alias — existing callers may import this name.
CONTEXT_AWARE_REQUIREMENTS = SCENARIO_GATED_REQUIREMENTS


def _max_contiguous_run(labels: np.ndarray, target: str) -> int:
    """Return the length of the longest contiguous run of `target`."""
    is_target = labels == target
    if not is_target.any():
        return 0
    padded = np.concatenate([[False], is_target, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def passes_plausibility_check(device_df: pd.DataFrame, attack_name: str) -> bool:
    """Check if a single device passes the realism gate for the given attack.

    Args:
        device_df: DataFrame rows for a single device (must have Scenario_Label)
        attack_name: name of the attack being evaluated

    Returns:
        True if the device has sufficient data for meaningful evaluation
    """
    # Tier 1: hard floor — enough rows for train/test split + windowing.
    if len(device_df) < HARD_FLOOR:
        return False

    # Tier 2: scenario-gated attacks require a contiguous segment of the
    # target driving scenario. Covers the 6 context-aware attacks and
    # eventual_stop (Cruising, 20 consecutive messages).
    req = SCENARIO_GATED_REQUIREMENTS.get(attack_name)
    if req is not None:
        target_scenario, min_length = req
        run = _max_contiguous_run(device_df["Scenario_Label"].values,
                                  target_scenario)
        if run < min_length:
            return False

    # Tier 3 (implicit): 18 simple attacks only need the hard floor.
    return True


def apply_hard_floor_gate(df: pd.DataFrame) -> tuple:
    """Filter devices by the hard floor only (L >= HARD_FLOOR).

    Use for mixed-attack evaluation where the per-attack context-aware
    checks don't apply — each device contributes to whatever attacks
    its data supports, and per-attack gating would over-filter.

    Args:
        df: DataFrame with device_id column

    Returns:
        (filtered_df, gate_log) with same format as apply_plausibility_check
    """
    has_sites = "_source_id" in df.columns
    site_stats = {}
    pass_devices = set()
    n_failed = 0

    for did, group in df.groupby("device_id"):
        site = group["_source_id"].iloc[0] if has_sites else "all"
        if site not in site_stats:
            site_stats[site] = {"pass": 0, "fail": 0}

        if len(group) >= HARD_FLOOR:
            pass_devices.add(did)
            site_stats[site]["pass"] += 1
        else:
            n_failed += 1
            site_stats[site]["fail"] += 1

    df_filtered = df[df["device_id"].isin(pass_devices)].reset_index(drop=True)

    gate_log = {
        "passed": len(pass_devices),
        "failed": n_failed,
        "by_site": site_stats,
    }
    return df_filtered, gate_log


def apply_plausibility_check(df: pd.DataFrame, attack_name: str) -> tuple:
    """Filter a multi-device DataFrame by the realism gate.

    Args:
        df: DataFrame with device_id and Scenario_Label columns
        attack_name: attack being evaluated

    Returns:
        (filtered_df, gate_log) where gate_log = {
            "passed": int, "failed": int,
            "by_site": {site: {"pass": n, "fail": n}, ...}
        }
    """
    has_sites = "_source_id" in df.columns
    site_stats = {}
    pass_devices = set()
    n_failed = 0

    for did, group in df.groupby("device_id"):
        site = group["_source_id"].iloc[0] if has_sites else "all"
        if site not in site_stats:
            site_stats[site] = {"pass": 0, "fail": 0}

        if passes_plausibility_check(group, attack_name):
            pass_devices.add(did)
            site_stats[site]["pass"] += 1
        else:
            n_failed += 1
            site_stats[site]["fail"] += 1

    df_filtered = df[df["device_id"].isin(pass_devices)].reset_index(drop=True)

    gate_log = {
        "passed": len(pass_devices),
        "failed": n_failed,
        "by_site": site_stats,
    }
    return df_filtered, gate_log
