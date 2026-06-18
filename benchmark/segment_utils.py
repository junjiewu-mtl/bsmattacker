"""
Shared Segment Partitioning Utilities
======================================
Used by the injection engine (injection_engine.py) and the benchmark driver
(run_benchmark.py) to assign contiguous temporal segments to benign/attack
slots and inject attacks per-segment (episode-correct).
"""

import numpy as np
import pandas as pd


def assign_segments(
    df: pd.DataFrame,
    attack_names: list,
    benign_fraction: float,
    segment_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Assign contiguous temporal segments to benign or attack slots.

    Each device's timeline (sorted by timestamp) is split into contiguous
    segments of approximately ``segment_size`` messages.  Each segment is
    randomly assigned to benign (slot 0) or one of the attack types
    (slots 1..N) according to ``benign_fraction``.

    Vectorized: uses cumcount + integer division instead of per-device loops.

    Returns
    -------
    np.ndarray
        Integer array aligned with ``df.index`` where 0 = benign and
        1..N maps to the corresponding entry in ``attack_names``.
    """
    n_attacks = len(attack_names)
    atk_frac = (1.0 - benign_fraction) / n_attacks
    slot_probs = [benign_fraction] + [atk_frac] * n_attacks

    # Vectorized segment assignment: row position within each device
    device_ids = df["device_id"].values
    dev_codes, dev_uniques = pd.factorize(device_ids)
    # Per-device cumulative row index
    row_within_dev = df.groupby("device_id").cumcount().values
    # Segment ID = row_within_dev // segment_size, made unique per device
    seg_within_dev = row_within_dev // segment_size
    # Create a global unique segment ID: (device_code, seg_within_dev)
    # Use a large multiplier to avoid collisions
    max_seg = seg_within_dev.max() + 1
    global_seg_id = dev_codes * max_seg + seg_within_dev

    # Get unique segment IDs and assign slots
    unique_segs = np.unique(global_seg_id)
    n_unique = len(unique_segs)
    seg_slot_values = rng.choice(n_attacks + 1, size=n_unique, p=slot_probs)

    # Map global_seg_id -> slot via a lookup array
    seg_id_to_slot = np.zeros(global_seg_id.max() + 1, dtype=int)
    seg_id_to_slot[unique_segs] = seg_slot_values

    slots = seg_id_to_slot[global_seg_id]
    return slots


def get_segment_boundaries(
    df: pd.DataFrame,
    slots: np.ndarray,
    attack_names: list,
) -> dict:
    """Group contiguous segment row-indices by (device_id, attack_name).

    Vectorized: uses diff-based boundary detection instead of per-device loops.

    Returns
    -------
    dict
        ``{attack_name: [(device_id, df_index_array), ...]}`` listing every
        contiguous segment assigned to that attack, per device.
    """
    segments_by_attack = {name: [] for name in attack_names}

    device_ids = df["device_id"].values
    df_index = df.index.values

    for i, attack_name in enumerate(attack_names):
        slot_id = i + 1
        mask = slots == slot_id
        if not mask.any():
            continue

        # Get all indices and device_ids for this attack slot
        atk_indices = df_index[mask]
        atk_devices = device_ids[mask]

        # Detect boundaries: device changes OR index gaps > 1
        dev_breaks = np.where(atk_devices[:-1] != atk_devices[1:])[0] + 1
        idx_gaps = np.where(np.diff(atk_indices) > 1)[0] + 1
        all_breaks = np.unique(np.concatenate([dev_breaks, idx_gaps]))

        # Split into runs
        runs = np.array_split(np.arange(len(atk_indices)), all_breaks)
        for run_positions in runs:
            if len(run_positions) == 0:
                continue
            run_indices = atk_indices[run_positions]
            did = atk_devices[run_positions[0]]
            segments_by_attack[attack_name].append((did, run_indices))

    return segments_by_attack
