"""
BSMAttacker: Context-Aware Attack Injection Library for V2X BSM Data
=====================================================================

This library provides a modular framework for injecting realistic, context-aware
attacks into V2X Basic Safety Message (BSM) datasets. It supports 28 attack
types, designed for different driving scenarios:

Original Attacks (Context-Aware):
 1. Position Pullthrough At Stop: False position attacks during stationary scenarios
 2. Lateral Drift At Turning: Aggressive position spoofing during turning scenarios
 3. False Deceleration Cruising: False sudden deceleration during cruising scenarios
 4. Slow Position Drift: Gradual position drift attacks across all scenarios
 5. Phantom Acceleration: False aggressive launch from stationary
 6. Heading Lock: Frozen heading during turns to hide turn maneuver

VeReMi-Equivalent Attacks (For SOTA Comparison):
 7. Constant Position: Frozen GPS coordinates (VeReMi Type 1)
 8. Constant Position Offset: Fixed spatial displacement (VeReMi Type 2)
 9. Random Position: Random position per timestep (VeReMi Type 3)
10. Random Position Offset: Random offset per timestep (VeReMi Type 4)
11. Constant Speed: Frozen speed value (VeReMi speed malfunction)
12. Random Speed: Random speed per timestep (VeReMi speed malfunction)
13. Eventual Stop: Gradual speed reduction to zero (VeReMi Type 16)
14. Data Replay: Replay of historical BSM sequences (VeReMi Extension)

VASP-Inspired Attacks (Expansion):
15. Constant Speed Offset: Fixed proportional speed bias (VeReMi A6 / VASP speed/ConstantOffset)
16. Random Speed Offset: Per-message proportional speed noise (VeReMi A8 / VASP speed/RandomOffset)
17. Constant Heading: Frozen heading value (VASP heading/Constant)
18. Constant Heading Offset: Fixed angular heading bias (VASP heading/ConstantOffset)
19. Random Heading: Random heading per message (VASP heading/Random)
20. Random Heading Offset: Per-message heading perturbation (VASP heading/RandomOffset)
21. Opposite Heading: 180-degree heading reversal (VASP heading/Opposite)
22. Perpendicular Heading: 90-degree heading rotation (VASP heading/Perpendicular)
23. Constant Acceleration: Frozen accel values (VASP acceleration/Constant)
24. Random Acceleration: Random accel per message (VASP acceleration/Random)
25. Random Acceleration Offset: Per-message accel perturbation (VASP acceleration/RandomOffset)
26. Ghost Vehicle: Phantom vehicle fabrication (VASP position/ghost_vehicle)
27. False Yield at Intersection: Low-speed/yield spoof for IMA-style situations
28. Speed Limit Violation: Posted-speed-limit spoof requiring speed-limit enrichment

Supports both legacy and extended CSV column formats.

Authors: Junjie Wu, Benjamin C. M. Fung, Hanbo Yu, Natalia Stakhanova
Date: January 2026, Updated February 2026
License: Apache-2.0
"""

from .base import (
    BaseAttacker,
    detect_csv_format,
    get_column_name,
    normalize_columns,
    COLUMN_ALIASES,
)
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

# VASP-inspired attacks (position — ghost vehicle)
from .ghost_vehicle import GhostVehicleAttacker

# safety-app-aligned additions
from .false_yield_at_intersection import FalseYieldAtIntersectionAttacker
from .speed_limit_violation import SpeedLimitViolationAttacker

from .pipeline import AttackPipeline

__version__ = "1.0.0"
__all__ = [
    # Base
    "BaseAttacker",
    "detect_csv_format",
    "get_column_name",
    "normalize_columns",
    "COLUMN_ALIASES",
    # Original attacks (IEEE TITS-style names)
    "PositionPullthroughAtStopAttacker",
    "LateralDriftAtTurningAttacker",
    "FalseDecelerationCruisingAttacker",
    "SlowPositionDriftAttacker",
    "PhantomAccelerationAttacker",
    "HeadingLockAttacker",
    # VeReMi-equivalent attacks
    "ConstantPositionAttacker",
    "ConstantPositionOffsetAttacker",
    "RandomPositionAttacker",
    "RandomPositionOffsetAttacker",
    "ConstantSpeedAttacker",
    "RandomSpeedAttacker",
    "EventualStopAttacker",
    "DataReplayAttacker",
    # VASP-inspired attacks (speed offsets)
    "ConstantSpeedOffsetAttacker",
    "RandomSpeedOffsetAttacker",
    # VASP-inspired attacks (heading)
    "ConstantHeadingAttacker",
    "ConstantHeadingOffsetAttacker",
    "RandomHeadingAttacker",
    "RandomHeadingOffsetAttacker",
    "OppositeHeadingAttacker",
    "PerpendicularHeadingAttacker",
    # VASP-inspired attacks (acceleration)
    "ConstantAccelerationAttacker",
    "RandomAccelerationAttacker",
    "RandomAccelerationOffsetAttacker",
    # VASP-inspired attacks (position)
    "GhostVehicleAttacker",
    # safety-app-aligned additions
    "FalseYieldAtIntersectionAttacker",
    "SpeedLimitViolationAttacker",
    # Pipeline
    "AttackPipeline",
]
