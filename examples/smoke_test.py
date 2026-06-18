#!/usr/bin/env python3
"""Smoke test: import the library and confirm the eleven paper attacks are registered.

Run:  python examples/smoke_test.py
"""
import pathlib
import sys

# Make the repo root importable when run directly (before `pip install -e .`).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bsm_attacker import AttackPipeline

# The eleven safety-application attacks evaluated in the paper.
PAPER_ATTACKS = [
    "constant_position_offset", "constant_speed", "eventual_stop",          # VeReMi Ext.
    "ghost_vehicle", "opposite_heading", "perpendicular_heading",            # VASP
    "constant_acceleration",
    "slow_position_drift", "false_deceleration_cruising",                    # this paper
    "position_pullthrough_at_stop", "heading_lock",
]


def main() -> int:
    pipeline = AttackPipeline(random_seed=42, show_semantics_warning=False)
    registered = set(pipeline.attackers)
    print(f"AttackPipeline ready; {len(registered)} attackers registered.")

    missing = [a for a in PAPER_ATTACKS if a not in registered]
    if missing:
        print(f"FAIL: paper attacks not registered: {missing}")
        return 1
    print(f"OK: all {len(PAPER_ATTACKS)} paper attacks are registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
