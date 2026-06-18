# `bsm_attacker` — the attack-injection library

`bsm_attacker` injects labeled attacks into benign V2X Basic Safety Message (BSM)
trajectories. It is the data-generation half of BSMAttacker; the evaluation harness lives
in [`../benchmark/`](../benchmark/).

## Input schema

Each input is a pandas `DataFrame`, one row per BSM. Column names are resolved through
aliases (see `base.py`), so both lower-case and capitalized variants work. The fields used
are:

| Field | Example column names | Notes |
|---|---|---|
| Device id | `device_id`, `station_id` | One sender per id. |
| Timestamp | `timestamp`, `timestamp_s` | Seconds. |
| Latitude / longitude | `latitude`, `longitude` | Degrees. |
| Speed | `speed_mps` | m/s. |
| Heading | `heading_deg` | Degrees. |
| Longitudinal / lateral / vertical acceleration | `accel_long_mps2`, `accel_lat_mps2`, `accel_vert_mps2` | m/s². |
| Yaw rate | `yaw_rate_degs`, `yaw_rate_dps` | deg/s. |
| Scenario label | `Scenario_Label` | Optional; auto-derived from speed and yaw if absent. |

The four context-aware attacks need a `Scenario_Label` (`Cruising`, `Turning`,
`Stationary`, ...). If the column is missing, `AttackPipeline` derives it from speed and
yaw rate before injecting.

## API

```python
from bsm_attacker import AttackPipeline

pipeline = AttackPipeline(random_seed=42)

# One attack -> a frame with an added Is_Attack column (1 = attacked row).
attacked = pipeline.inject_single_attack(df, "slow_position_drift")

# Many attacks -> a dict of {attack_name: attacked_frame}.
results = pipeline.inject_all_attacks(
    df, attack_configs={"constant_speed": {"family": "veremi_equivalent"}})

# Persist the attacked datasets.
pipeline.save_attacked_datasets(results, "./output")
```

Every attacker subclasses `BaseAttacker` (`base.py`) and implements `inject_attack`. The
VeReMi Extension sensor-error model is in `noise_model.py`; geodetic helpers are in
`geo_constants.py`.

## Attack families

- **VeReMi / VeReMi Extension**: constant and random position and speed attacks, eventual
  stop.
- **VASP**: heading, speed-offset, acceleration, and ghost-vehicle attacks.
- **Context-aware (this paper)**: slow position drift, false hard-brake in cruising,
  phantom forward-roll at stop, frozen heading at turn. Each fires only inside the driving
  scenario it targets.

The eleven attacks evaluated in the paper are listed in the top-level
[`README.md`](../README.md). The library implements additional attacks beyond those eleven.

## Adding an attack

Subclass `BaseAttacker`, implement `inject_attack(self, df, **kwargs)` so it returns the
frame with an `Is_Attack` column set, and register the class in `pipeline.py`.
