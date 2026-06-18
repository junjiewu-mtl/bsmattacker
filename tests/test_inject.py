"""Minimal functional tests: the pipeline injects an attack and labels rows.

Run:  python -m pytest tests/ -q       (or)      python tests/test_inject.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd

# Make the repo root importable when run directly (before `pip install -e .`).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bsm_attacker import AttackPipeline

PAPER_ATTACKS = [
    "constant_position_offset", "constant_speed", "eventual_stop",
    "ghost_vehicle", "opposite_heading", "perpendicular_heading",
    "constant_acceleration", "slow_position_drift",
    "false_deceleration_cruising", "position_pullthrough_at_stop", "heading_lock",
]


def _toy_trace(n: int = 60) -> pd.DataFrame:
    """A short synthetic cruising trace anchored at (0, 0). No real location."""
    t = np.arange(n) * 0.1
    return pd.DataFrame({
        "device_id": "VEH001",
        "timestamp": t,
        "latitude": 0.0 + 1e-6 * np.arange(n),
        "longitude": 0.0 + 1.9e-5 * np.arange(n),
        "speed_mps": 15.0 + 0.2 * np.sin(t),
        "heading_deg": 90.0 + 0.1 * np.cos(t),
        "accel_long_mps2": 0.1 * np.cos(t),
        "accel_lat_mps2": 0.01 * np.sin(t),
        "accel_vert_mps2": 0.0,
        "yaw_rate_degs": 0.1 * np.sin(t),
        "msg_count": np.arange(n),
    })


def test_paper_attacks_registered():
    pipeline = AttackPipeline(random_seed=42, show_semantics_warning=False)
    missing = [a for a in PAPER_ATTACKS if a not in pipeline.attackers]
    assert not missing, f"paper attacks not registered: {missing}"


def test_inject_constant_speed_labels_rows():
    pipeline = AttackPipeline(random_seed=42, show_semantics_warning=False)
    attacked = pipeline.inject_single_attack(_toy_trace(), "constant_speed")
    assert "Is_Attack" in attacked.columns
    assert attacked["Is_Attack"].sum() > 0, "no rows were flagged as attacked"


def test_inject_context_aware_runs():
    pipeline = AttackPipeline(random_seed=42, show_semantics_warning=False)
    attacked = pipeline.inject_single_attack(_toy_trace(), "slow_position_drift")
    assert "Is_Attack" in attacked.columns


if __name__ == "__main__":
    test_paper_attacks_registered()
    test_inject_constant_speed_labels_rows()
    test_inject_context_aware_runs()
    print("OK: all functional tests passed.")
