#!/usr/bin/env python3
"""
Attack-injection and feature engine for the V2X BSM benchmark.

Provides the corpus loader, scenario labeler, attack-injection routines
(per-segment and vectorized), derived-feature engineering, the sliding-window
sequence dataset, and the evaluation helpers consumed by ``run_benchmark.py``.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix,
)
import yaml

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
_MAIN_REPO = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from bsm_attacker.pipeline import AttackPipeline
from benchmark.segment_utils import assign_segments, get_segment_boundaries


# ======================================================================
# Configuration
# ======================================================================

def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = SCRIPT_DIR / "configs" / "benchmark.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ======================================================================
# Data loading
# ======================================================================

def load_corpus(cfg: dict, corpus_mode: str = "core") -> pd.DataFrame:
    """Load and pool canonical parquet files.

    Args:
        corpus_mode: "core" restricts to the primary (the primary sites) sources;
            any other value pools all sources in the manifest.

    Returns:
        Pooled DataFrame with unique device_id (prefixed with source_id).
    """
    canon_dir = _MAIN_REPO / cfg["corpus"]["canonical_dir"]
    manifest_path = _MAIN_REPO / cfg["corpus"]["manifest"]
    if not manifest_path.exists():
        # Fallback to PROJECT_ROOT (non-worktree case)
        canon_dir = PROJECT_ROOT / cfg["corpus"]["canonical_dir"]
        manifest_path = PROJECT_ROOT / cfg["corpus"]["manifest"]
    manifest = pd.read_csv(manifest_path)

    if corpus_mode == "core":
        mask = manifest["source_id"].str.startswith("T1")
        manifest = manifest[mask]
        print(f"Primary corpus: {len(manifest)} sources")
    else:
        print(f"Full corpus: {len(manifest)} sources")

    frames = []
    for _, row in manifest.iterrows():
        src_id = row["source_id"]
        parquet_path = canon_dir / f"{src_id}.parquet"
        if not parquet_path.exists():
            print(f"  [WARN] Missing: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        # Prefix device_id to avoid collisions across sources
        df["device_id"] = src_id + "_" + df["device_id"].astype(str)
        df["_source_id"] = src_id
        frames.append(df)
        print(f"  Loaded {src_id}: {len(df):,} rows, "
              f"{df['device_id'].nunique()} devices")

    pooled = pd.concat(frames, ignore_index=True)

    # Apply column bridge
    bridge = cfg.get("column_bridge", {})
    if bridge:
        pooled = pooled.rename(columns=bridge)
        print(f"  Column bridge applied: {bridge}")

    print(f"  Pooled: {len(pooled):,} rows, "
          f"{pooled['device_id'].nunique()} unique devices")
    return pooled


# ======================================================================
# Scenario labeling
# ======================================================================

SPEED_THRESHOLD = 0.5      # m/s — matches BSMAttacker pipeline
YAW_RATE_THRESHOLD = 5.0   # deg/s — standardized across all labeling

# Duration thresholds for Stationary sub-labels
STATIONARY_BRIEF_MAX_S = 5.0    # < 5 s  → Stationary_Brief (stop signs)
STATIONARY_WAIT_MAX_S = 60.0    # 5–60 s → Stationary_Wait  (traffic lights)
                                 # > 60 s → Stationary_Parked (parking)

# Sustained-condition smoothing for the Turning label: 20–50% of single-sample
# Turning labels can be noise spikes. Requiring k consecutive samples above
# threshold per device suppresses noise-induced false Turning without
# over-smoothing clean sites. Default k=1 preserves the instantaneous behavior;
# set k=3 (300 ms at 10 Hz) for noise-robust labeling.
SUSTAINED_K_DEFAULT = 1


def label_scenarios(df: pd.DataFrame, sustained_k: int = SUSTAINED_K_DEFAULT) -> pd.DataFrame:
    """Add Scenario_Label with duration-aware Stationary sub-labels.

    Stationary messages are further split by the duration of each contiguous
    stationary segment per device:
      - Stationary_Brief  (< 5 s):   stop signs, momentary pauses
      - Stationary_Wait   (5–60 s):  traffic lights, intersection waits
      - Stationary_Parked (> 60 s):  parking, pre-drive warmup

    The Turning label may require a sustained yaw-rate excursion.
    With ``sustained_k=1`` (default), any sample with ``|yaw| > threshold``
    is labeled Turning (instantaneous). With ``sustained_k>=2``, Turning
    applies only to samples inside a run of ``>=sustained_k`` contiguous
    hot samples on the same device. This suppresses noise-induced isolated
    spikes without requiring threshold tuning.

    Parameters
    ----------
    df : DataFrame
        BSM data with ``speed_mps``, ``yaw_rate_degs``, ``device_id``.
    sustained_k : int, default 1
        Minimum run length for Turning. 1 = instantaneous;
        3 = 300 ms at 10 Hz BSM rate.
    """
    df = df.copy()

    # --- Step 1: base labels --------------------------------------------
    df["Scenario_Label"] = "Cruising"
    stationary_mask = df["speed_mps"] < SPEED_THRESHOLD
    df.loc[stationary_mask, "Scenario_Label"] = "Stationary"
    yaw_col = "yaw_rate_degs"
    raw_hot = df[yaw_col].abs() > YAW_RATE_THRESHOLD

    if sustained_k <= 1:
        turning_mask = raw_hot
    else:
        # Run-length smoothing: Turning only when k+ consecutive hot samples
        # on the same device. Labels the WHOLE sustained run, not just the
        # confirmation tail.
        dev_change = df["device_id"].ne(df["device_id"].shift())
        state_change = raw_hot.ne(raw_hot.shift())
        run_id = (state_change | dev_change).cumsum()
        run_size = raw_hot.groupby(run_id).transform("size")
        turning_mask = raw_hot & (run_size >= sustained_k)

    df.loc[turning_mask, "Scenario_Label"] = "Turning"

    # --- Step 2: duration-aware Stationary sub-labels -------------------
    # Re-derive stationary_mask after Turning override (Turning takes precedence)
    stationary_mask = df["Scenario_Label"] == "Stationary"

    if stationary_mask.any():
        # Vectorised run-length grouping: new group whenever stationary
        # status changes OR device_id changes
        _is_stat = stationary_mask.astype(int)
        _shifted = _is_stat.ne(_is_stat.shift()) | df["device_id"].ne(df["device_id"].shift())
        _stat_grp = _shifted.cumsum()

        ts_col = "timestamp"
        stat_df = df.loc[stationary_mask, [ts_col, "device_id"]].copy()
        stat_df["_stat_grp"] = _stat_grp[stationary_mask]

        # Vectorised: compute duration per segment group
        seg_stats = stat_df.groupby("_stat_grp").agg(
            ts_min=(ts_col, "first"),
            ts_max=(ts_col, "last"),
            count=(ts_col, "size"),
        )

        if pd.api.types.is_numeric_dtype(df[ts_col]):
            seg_stats["duration_s"] = seg_stats["ts_max"] - seg_stats["ts_min"]
        else:
            seg_stats["duration_s"] = (seg_stats["count"] - 1) * 0.1

        # Map duration to sub-label
        seg_stats["sublabel"] = "Stationary_Brief"
        seg_stats.loc[
            seg_stats["duration_s"] >= STATIONARY_BRIEF_MAX_S, "sublabel"
        ] = "Stationary_Wait"
        seg_stats.loc[
            seg_stats["duration_s"] > STATIONARY_WAIT_MAX_S, "sublabel"
        ] = "Stationary_Parked"

        # Map back to original DataFrame via _stat_grp
        sublabel_map = seg_stats["sublabel"]
        df.loc[stationary_mask, "Scenario_Label"] = (
            _stat_grp[stationary_mask].map(sublabel_map).values
        )

    counts = df["Scenario_Label"].value_counts()
    print(f"  Scenario labels: {counts.to_dict()}")
    return df


# ======================================================================
# Attack injection (episode-correct, segment-based)
# ======================================================================

# Column mapping: canonical → attacker CamelCase
CANONICAL_TO_ATTACKER = {
    "timestamp": "Tx_Timestamp",
    "device_id": "Device_ID",
    "latitude": "Latitude_deg",
    "longitude": "Longitude_deg",
    "speed_mps": "Speed_mps",
    "heading_deg": "Heading_deg",
    "accel_long_mps2": "Accel_Long_mps2",
    "accel_lat_mps2": "Accel_Lat_mps2",
    "accel_vert_mps2": "Accel_Vert_mps2",
    "yaw_rate_degs": "Yaw_Rate",
}
ATTACKER_TO_CANONICAL = {v: k for k, v in CANONICAL_TO_ATTACKER.items()}

FEATURE_COLS = [
    "latitude", "longitude", "speed_mps", "heading_deg",
    "accel_long_mps2", "accel_lat_mps2", "accel_vert_mps2", "yaw_rate_degs",
]


def _injection_worker(args):
    """Worker for multiprocess injection. Must be module-level for pickling."""
    batch, atk_name, atk_params, base_seed, df_att_chunk = args
    pipeline = AttackPipeline(random_seed=base_seed)
    for a in pipeline.attackers.values():
        a.verbose = False
    results = []
    for j, (dev_id, seg_idx) in enumerate(batch):
        seg_rows = df_att_chunk.loc[seg_idx].copy()
        seg_rows["_orig_idx"] = seg_idx
        try:
            np.random.seed(base_seed + j)
            seg_attacked = pipeline.inject_single_attack(
                seg_rows, atk_name, auto_preprocess=False,
                **atk_params,
            )
            results.append((dev_id, seg_idx, seg_attacked, j))
        except Exception:
            results.append(None)
    return results


# ======================================================================
# Vectorized attack injection
# ======================================================================

# Attacks that produce MORE rows than input (row-expanding).
# These cannot be bulk-vectorized and must keep the per-segment loop.
ROW_EXPANDING_ATTACKS = {"data_replay", "ghost_vehicle"}

# All other 22 attacks are "simple": same row count in/out, modify columns
# in-place. Grouped by which canonical columns they touch and how.
#
# The vectorized path replicates each attacker's inject_attack() logic
# but operates on the full attack-assigned bulk in one pass, eliminating
# 12K+ per-segment DataFrame copies, column renames, and function calls.


def _select_attack_rows_per_segment(
    seg_map_entries: list,
    attack_ratio: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Select which rows within each segment to actually attack.

    Each attacker internally does:
        n_attack = max(1, int(len(vehicle_indices) * attack_ratio))
        attack_indices = np.random.choice(vehicle_indices, n_attack, replace=False)

    This function replicates that logic across all segments at once,
    returning a flat numpy array of all selected row indices.
    """
    selected = []
    for _dev_id, seg_indices in seg_map_entries:
        n = len(seg_indices)
        n_attack = max(1, int(n * attack_ratio))
        chosen = rng.choice(seg_indices, size=n_attack, replace=False)
        selected.append(chosen)
    if not selected:
        return np.array([], dtype=np.intp)
    return np.concatenate(selected)


def _select_contiguous_attack_rows_per_segment(
    seg_map_entries: list,
    attack_ratio: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Select contiguous runs from the START of each segment.

    Used by liar_at_light, intersection_bluff, brake_bluff which attack
    contiguous windows from each segment, not random rows.
    """
    selected = []
    for _dev_id, seg_indices in seg_map_entries:
        n = len(seg_indices)
        n_attack = max(1, int(n * attack_ratio))
        selected.append(seg_indices[:n_attack])
    if not selected:
        return np.array([], dtype=np.intp)
    return np.concatenate(selected)


def _vectorized_inject_simple(
    df: pd.DataFrame,
    attack_name: str,
    attack_ratio: float,
    attack_pos: np.ndarray,
    seg_ids: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    n_segs: int,
    seed: int,
    cartesian: bool = False,
) -> int:
    """Bulk-vectorized injection for simple (same-row-count) attacks.

    Modifies ``df`` IN PLACE (is_attack, attack_type, and feature columns).
    When ``cartesian=True``, position offsets are applied in raw metres
    (for SUMO/VeReMi Cartesian corpora) instead of GPS degree conversion.
    Returns the number of rows actually attacked.

    Parameters
    ----------
    attack_pos : array of int
        Positional (iloc) indices into ``df`` that are attack-assigned.
    seg_ids : array of int, same length as ``attack_pos``
        Segment ID for each position in ``attack_pos``.
    seg_starts, seg_ends : arrays of int, length ``n_segs``
        ``attack_pos[seg_starts[i]:seg_ends[i]]`` gives the positions
        for segment *i*.
    n_segs : int
        Number of segments.

    The logic below mirrors each attacker class's inject_attack() method
    but operates on the entire attack-assigned population at once.
    """
    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Helpers: fast bulk row selection using pre-built segment arrays
    # ------------------------------------------------------------------

    def _bulk_random_select():
        """Select random rows per segment (same as _select_attack_rows_per_segment)."""
        selected = np.zeros(len(attack_pos), dtype=bool)
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            selected[s + chosen] = True
        return attack_pos[selected]

    def _bulk_contiguous_select():
        """Select contiguous rows from start of each segment."""
        selected = np.zeros(len(attack_pos), dtype=bool)
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            selected[s:s + n_atk] = True
        return attack_pos[selected]

    # Pre-fetch underlying numpy arrays for fast iloc access.
    # Ensure they are writable (parquet-backed arrays may be read-only).
    def _writable(col):
        arr = df[col].values
        if not arr.flags.writeable:
            arr.flags.writeable = True
        return arr

    # Pre-make is_attack and attack_type writable for fast numpy writes
    _is_attack_arr = _writable("is_attack")
    # attack_type may be Arrow/string dtype — convert to plain numpy object array
    if not isinstance(df["attack_type"].values, np.ndarray):
        df["attack_type"] = df["attack_type"].astype(object)
    _attack_type_arr = df["attack_type"].values
    if not _attack_type_arr.flags.writeable:
        _attack_type_arr.flags.writeable = True

    def _mark_attacked(idx):
        """Set is_attack and attack_type on selected positional indices."""
        _is_attack_arr[idx] = 1
        _attack_type_arr[idx] = attack_name

    lat_vals = _writable("latitude")
    lon_vals = _writable("longitude")

    # ------------------------------------------------------------------
    # Dispatch by attack name
    # ------------------------------------------------------------------

    # === POSITION ATTACKS =============================================

    if attack_name == "constant_position":
        # Freeze lat/lon to first observed position per segment.
        # Per-segment constant: pick frozen lat/lon from first row of segment.
        selected = np.zeros(len(attack_pos), dtype=bool)
        frozen_lats = np.empty(len(attack_pos))
        frozen_lons = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            first_pos = attack_pos[s]
            fl = lat_vals[first_pos]
            fo = lon_vals[first_pos]
            for c in chosen:
                selected[s + c] = True
                frozen_lats[s + c] = fl
                frozen_lons[s + c] = fo
        attacked_idx = attack_pos[selected]
        lat_vals[attacked_idx] = frozen_lats[selected]
        lon_vals[attacked_idx] = frozen_lons[selected]
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "constant_position_offset":
        from bsm_attacker.geo_constants import DEG_LAT_TO_M, deg_lon_to_m
        meters_per_degree_lat = DEG_LAT_TO_M
        meters_per_degree_lon = deg_lon_to_m(df["latitude"].median())
        # Per-segment constant offset + direction
        selected = np.zeros(len(attack_pos), dtype=bool)
        delta_lat_arr = np.empty(len(attack_pos))
        delta_lon_arr = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            offset_m = rng.uniform(100, 500)  # VeReMi Type 2: 100-500m (Kamel et al. 2020)
            direction = rng.uniform(0, 2 * np.pi)
            if cartesian:
                dl = offset_m * np.cos(direction)
                do = offset_m * np.sin(direction)
            else:
                dl = (offset_m * np.cos(direction)) / meters_per_degree_lat
                do = (offset_m * np.sin(direction)) / meters_per_degree_lon
            for c in chosen:
                selected[s + c] = True
                delta_lat_arr[s + c] = dl
                delta_lon_arr[s + c] = do
        attacked_idx = attack_pos[selected]
        lat_vals[attacked_idx] += delta_lat_arr[selected]
        lon_vals[attacked_idx] += delta_lon_arr[selected]
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "random_position":
        from bsm_attacker.geo_constants import DEG_LAT_TO_M, deg_lon_to_m
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            offsets_m = rng.uniform(5, 50, size=n)
            directions = rng.uniform(0, 2 * np.pi, size=n)
            if cartesian:
                lat_vals[attacked_idx] += offsets_m * np.cos(directions)
                lon_vals[attacked_idx] += offsets_m * np.sin(directions)
            else:
                lat_vals[attacked_idx] += (offsets_m * np.cos(directions)) / DEG_LAT_TO_M
                lon_vals[attacked_idx] += (offsets_m * np.sin(directions)) / deg_lon_to_m(lat_vals[attacked_idx].mean())
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "random_position_offset":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            offsets_m = rng.uniform(5, 30, size=n)
            directions = rng.uniform(0, 2 * np.pi, size=n)
            north_m = offsets_m * np.cos(directions)
            east_m = offsets_m * np.sin(directions)
            if cartesian:
                lat_vals[attacked_idx] += north_m
                lon_vals[attacked_idx] += east_m
            else:
                from bsm_attacker.geo_constants import EARTH_RADIUS_M
                R = EARTH_RADIUS_M
                lats = lat_vals[attacked_idx].copy()
                lat_vals[attacked_idx] += north_m / R * (180 / np.pi)
                lon_vals[attacked_idx] += (
                    east_m / R * (180 / np.pi) / np.cos(lats * np.pi / 180)
                )
            _mark_attacked(attacked_idx)
        return n

    # === SPEED ATTACKS ================================================

    elif attack_name == "constant_speed":
        # Per-segment constant: one frozen_speed per segment
        speed_vals = _writable("speed_mps")
        selected = np.zeros(len(attack_pos), dtype=bool)
        frozen_speeds = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            fs = rng.uniform(0, 40)
            for c in chosen:
                selected[s + c] = True
                frozen_speeds[s + c] = fs
        attacked_idx = attack_pos[selected]
        speed_vals[attacked_idx] = frozen_speeds[selected]
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "random_speed":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            _writable("speed_mps")[attacked_idx] = rng.uniform(0, 40, size=n)
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "constant_speed_offset":
        # Per-segment constant offset with random sign
        speed_vals = _writable("speed_mps")
        selected = np.zeros(len(attack_pos), dtype=bool)
        offset_arr = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            offset = rng.uniform(2.0, 10.0)
            if rng.random() < 0.5:
                offset = -offset
            for c in chosen:
                selected[s + c] = True
                offset_arr[s + c] = offset
        attacked_idx = attack_pos[selected]
        speed_vals[attacked_idx] = np.clip(
            speed_vals[attacked_idx] + offset_arr[selected], 0, None
        )
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "random_speed_offset":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            offsets = rng.uniform(-10.0, 10.0, size=n)
            speed_vals = _writable("speed_mps")
            speed_vals[attacked_idx] = np.clip(
                speed_vals[attacked_idx] + offsets, 0, None
            )
            _mark_attacked(attacked_idx)
        return n

    # === HEADING ATTACKS ==============================================

    elif attack_name == "constant_heading":
        # Per-segment constant frozen heading
        heading_vals = _writable("heading_deg")
        selected = np.zeros(len(attack_pos), dtype=bool)
        frozen_headings = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            fh = rng.uniform(0, 360)
            for c in chosen:
                selected[s + c] = True
                frozen_headings[s + c] = fh
        attacked_idx = attack_pos[selected]
        heading_vals[attacked_idx] = frozen_headings[selected]
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "constant_heading_offset":
        # Per-segment constant offset with random sign
        heading_vals = _writable("heading_deg")
        selected = np.zeros(len(attack_pos), dtype=bool)
        offset_arr = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            offset_deg = rng.uniform(15, 90)
            if rng.random() < 0.5:
                offset_deg = -offset_deg
            for c in chosen:
                selected[s + c] = True
                offset_arr[s + c] = offset_deg
        attacked_idx = attack_pos[selected]
        heading_vals[attacked_idx] = (
            heading_vals[attacked_idx] + offset_arr[selected]
        ) % 360
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "random_heading":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            _writable("heading_deg")[attacked_idx] = rng.uniform(0, 360, size=n)
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "random_heading_offset":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            offsets = rng.uniform(-90, 90, size=n)
            heading_vals = _writable("heading_deg")
            heading_vals[attacked_idx] = (
                heading_vals[attacked_idx] + offsets
            ) % 360
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "opposite_heading":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            heading_vals = _writable("heading_deg")
            heading_vals[attacked_idx] = (
                heading_vals[attacked_idx] + 180
            ) % 360
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "perpendicular_heading":
        # Per-segment constant rotation (+90 or -90)
        heading_vals = _writable("heading_deg")
        selected = np.zeros(len(attack_pos), dtype=bool)
        rotation_arr = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            rotation = 90 if rng.random() < 0.5 else -90
            for c in chosen:
                selected[s + c] = True
                rotation_arr[s + c] = rotation
        attacked_idx = attack_pos[selected]
        heading_vals[attacked_idx] = (
            heading_vals[attacked_idx] + rotation_arr[selected]
        ) % 360
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    # === ACCELERATION ATTACKS =========================================

    elif attack_name == "constant_acceleration":
        # Per-segment constant frozen long/lat accel
        along_vals = _writable("accel_long_mps2")
        alat_vals = _writable("accel_lat_mps2")
        selected = np.zeros(len(attack_pos), dtype=bool)
        frozen_long = np.empty(len(attack_pos))
        frozen_lat = np.empty(len(attack_pos))
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            n = e - s
            n_atk = max(1, int(n * attack_ratio))
            chosen = rng.choice(n, size=n_atk, replace=False)
            fl = rng.uniform(-5, 5)
            fla = rng.uniform(-3, 3)
            for c in chosen:
                selected[s + c] = True
                frozen_long[s + c] = fl
                frozen_lat[s + c] = fla
        attacked_idx = attack_pos[selected]
        along_vals[attacked_idx] = frozen_long[selected]
        alat_vals[attacked_idx] = frozen_lat[selected]
        _mark_attacked(attacked_idx)
        return len(attacked_idx)

    elif attack_name == "random_acceleration":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            _writable("accel_long_mps2")[attacked_idx] = rng.uniform(-8, 8, size=n)
            _writable("accel_lat_mps2")[attacked_idx] = rng.uniform(-5, 5, size=n)
            _mark_attacked(attacked_idx)
        return n

    elif attack_name == "random_acceleration_offset":
        attacked_idx = _bulk_random_select()
        n = len(attacked_idx)
        if n > 0:
            _writable("accel_long_mps2")[attacked_idx] += rng.uniform(-4.0, 4.0, size=n)
            _writable("accel_lat_mps2")[attacked_idx] += rng.uniform(-3.0, 3.0, size=n)
            _mark_attacked(attacked_idx)
        return n

    # === CONTEXT-AWARE ATTACKS ========================================

    elif attack_name in ("liar_at_light", "position_pullthrough_at_stop"):
        # Co-spoof position + speed + accel + heading.
        # Adds bell-curve speed ramp to defeat pos_vel_consistency.
        scenario_vals = df["Scenario_Label"].values
        heading_vals = _writable("heading_deg")
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        if not cartesian:
            from bsm_attacker.geo_constants import EARTH_RADIUS_M
            R = EARTH_RADIUS_M
        total_attacked = 0
        speed_ramp_msgs = 5
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            # Pull-through targets 5-60 s red-light waits (Stationary_Wait).
            wait_mask = scenario_vals[seg_pos] == "Stationary_Wait"
            wait_pos = seg_pos[wait_mask]
            if len(wait_pos) == 0:
                continue
            n_attack = max(1, int(len(wait_pos) * attack_ratio))
            aidx = wait_pos[:n_attack]
            n_a = len(aidx)

            base_offset = rng.uniform(20, 50)
            peak_speed = rng.uniform(1.5, 3.0)
            gps_noise_std = 2.5

            # Bell-curve speed ramp for first N messages
            n_ramp = min(speed_ramp_msgs, n_a)
            ramp_profile = np.sin(np.linspace(0, np.pi, n_ramp))
            speed_bell = ramp_profile * peak_speed

            # Estimate dt
            dt = 0.1
            if n_a > 1:
                t0 = df["timestamp"].values[aidx[0]] if "timestamp" in df.columns else None
                t1 = df["timestamp"].values[aidx[1]] if t0 is not None else None
                if t0 is not None and t1 is not None:
                    _dt = abs(float(t1) - float(t0))
                    if _dt > 0:
                        dt = _dt

            # Acceleration from speed differences
            accel_bell = np.zeros(n_ramp)
            accel_bell[1:] = np.diff(speed_bell) / dt

            # Cumulative position from speed profile, scaled to base_offset
            cum_pos = np.cumsum(speed_bell * dt)
            pos_scale = base_offset / cum_pos[-1] if cum_pos[-1] > 0 else 1.0

            # Per-message offsets: ramp phase uses cum_pos, hold phase uses base_offset
            msg_offsets = np.full(n_a, base_offset)
            msg_offsets[:n_ramp] = cum_pos * pos_scale

            # Speed/accel co-spoofing
            speed_vals[aidx[:n_ramp]] = speed_bell
            if n_a > n_ramp:
                speed_vals[aidx[n_ramp:]] = 0.0
            accel_vals[aidx[:n_ramp]] = accel_bell
            if n_a > n_ramp:
                accel_vals[aidx[n_ramp:]] = 0.0

            # Heading: tilt toward offset direction + small noise
            offset_heading = heading_vals[aidx[0]]
            heading_vals[aidx] = offset_heading + rng.normal(0, 2.0, size=n_a)

            # Position offsets in offset_heading direction + GPS noise
            noise_north = rng.normal(0, gps_noise_std, size=n_a)
            noise_east = rng.normal(0, gps_noise_std, size=n_a)
            offset_north = msg_offsets * np.cos(np.radians(offset_heading)) + noise_north
            offset_east = msg_offsets * np.sin(np.radians(offset_heading)) + noise_east

            if cartesian:
                lat_vals[aidx] += offset_north
                lon_vals[aidx] += offset_east
            else:
                lats = lat_vals[aidx].copy()
                lat_vals[aidx] += offset_north / R * (180 / np.pi)
                lon_vals[aidx] += (
                    offset_east / R * (180 / np.pi) / np.cos(lats * np.pi / 180)
                )
            is_attack_vals[aidx] = 1
            attack_type_vals[aidx] = attack_name
            total_attacked += n_a
        return total_attacked

    elif attack_name in ("intersection_bluff", "lateral_drift_at_turning"):
        # Co-spoof position + heading tilt toward offset.
        scenario_vals = df["Scenario_Label"].values
        heading_vals = _writable("heading_deg")
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        if not cartesian:
            from bsm_attacker.geo_constants import EARTH_RADIUS_M
            R = EARTH_RADIUS_M
        total_attacked = 0
        heading_tilt_factor = 0.4
        max_heading_tilt = 5.0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            # Lateral-drift targets the turning phase.
            turn_mask = scenario_vals[seg_pos] == "Turning"
            turn_pos = seg_pos[turn_mask]
            if len(turn_pos) == 0:
                continue
            n_attack = max(1, int(len(turn_pos) * attack_ratio))
            aidx = turn_pos[:n_attack]
            n_a = len(aidx)

            direction = rng.choice([-1, 1])
            max_offset = rng.uniform(5, 15)

            progress = np.arange(1, n_a + 1, dtype=np.float64) / n_a
            offset_distance = max_offset * progress

            headings_seg = heading_vals[aidx].copy()
            perp_heading = headings_seg + (90 * direction)
            offset_north = offset_distance * np.cos(np.radians(perp_heading))
            offset_east = offset_distance * np.sin(np.radians(perp_heading))

            if cartesian:
                lat_vals[aidx] += offset_north
                lon_vals[aidx] += offset_east
            else:
                lats = lat_vals[aidx].copy()
                lat_vals[aidx] += offset_north / R * (180 / np.pi)
                lon_vals[aidx] += (
                    offset_east / R * (180 / np.pi) / np.cos(lats * np.pi / 180)
                )

            # v2.0: Heading tilt toward offset direction
            heading_tilt = direction * np.minimum(max_heading_tilt, offset_distance * heading_tilt_factor)
            heading_vals[aidx] = headings_seg + heading_tilt

            is_attack_vals[aidx] = 1
            attack_type_vals[aidx] = attack_name
            total_attacked += n_a
        return total_attacked

    elif attack_name in ("brake_bluff", "false_deceleration_cruising"):
        # Co-spoof speed + accel + position deceleration.
        scenario_vals = df["Scenario_Label"].values
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        heading_vals = df["heading_deg"].values
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        if not cartesian:
            from bsm_attacker.geo_constants import EARTH_RADIUS_M
            R = EARTH_RADIUS_M
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            cruise_mask = scenario_vals[seg_pos] == "Cruising"
            cruise_pos = seg_pos[cruise_mask]
            if len(cruise_pos) == 0:
                continue
            n_attack = max(1, int(len(cruise_pos) * attack_ratio))
            n_attack = min(n_attack, len(cruise_pos))
            aidx = cruise_pos[:n_attack]
            n_a = len(aidx)

            fake_decel = rng.uniform(-5.0, -3.0)
            initial_speed = speed_vals[aidx[0]]
            if pd.isna(initial_speed) or initial_speed < 1.0:
                continue

            dt = 0.1
            elapsed = np.arange(n_a, dtype=np.float64) * dt
            new_speeds = np.maximum(0, initial_speed + fake_decel * elapsed)

            speed_vals[aidx] = new_speeds
            accel_vals[aidx] = fake_decel

            # v2.0: Co-spoof position — offset backwards to match braking
            # Real position keeps cruising; spoofed should decelerate
            heading_rad = np.radians(heading_vals[aidx[0]])
            # Cumulative displacement under braking: sum(v*dt + 0.5*a*dt²)
            step_disp = np.maximum(0, new_speeds[:-1] * dt + 0.5 * fake_decel * dt ** 2)
            cum_brake_disp = np.zeros(n_a)
            cum_brake_disp[1:] = np.cumsum(step_disp)
            # Real cruise displacement: v0 * t
            real_cruise_disp = initial_speed * elapsed
            # Backward offset = how much less distance braking covers vs cruising
            backward_offset = real_cruise_disp - cum_brake_disp
            # Only apply from message 1 onwards (message 0 has no offset)
            off_n = -backward_offset * np.cos(heading_rad)
            off_e = -backward_offset * np.sin(heading_rad)

            if cartesian:
                lat_vals[aidx] += off_n
                lon_vals[aidx] += off_e
            else:
                lats = lat_vals[aidx].copy()
                lat_vals[aidx] += off_n / R * (180 / np.pi)
                lon_vals[aidx] += (
                    off_e / R * (180 / np.pi) / np.cos(lats * np.pi / 180)
                )

            is_attack_vals[aidx] = 1
            attack_type_vals[aidx] = attack_name
            total_attacked += n_a
        return total_attacked

    elif attack_name in ("stealth_drift", "slow_position_drift"):
        # Navigation Drift — FCW-targeted with speed co-spoof.
        # Drift rate 1-3 m/msg, max 30m, speed bias for plausibility.
        if not cartesian:
            from bsm_attacker.geo_constants import DEG_LAT_TO_M as _DLM
        scenario_vals = df["Scenario_Label"].values
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        speed_vals = _writable("speed_mps")
        ar1_rho = 0.95
        direction_innovation_std = 10.0
        max_cumulative = 30.0
        speed_bias_lo, speed_bias_hi = 0.5, 2.0
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            cruise_mask = scenario_vals[seg_pos] == "Cruising"
            cruise_pos = seg_pos[cruise_mask]
            n_seg = len(cruise_pos)
            if n_seg == 0:
                continue
            n_attack = max(1, int(n_seg * attack_ratio))
            if n_seg > n_attack:
                start = rng.randint(0, n_seg - n_attack + 1)
                aidx = cruise_pos[start:start + n_attack]
            else:
                aidx = cruise_pos
            n_a = len(aidx)

            # v3.0: increased drift rate (1.0–3.0 m/msg) for FCW impact
            drift_distances = rng.uniform(1.0, 3.0, size=n_a)

            # v2.0: AR(1) autocorrelated drift direction (ρ=0.95)
            # θ_t = ρ * θ_{t-1} + (1-ρ) * ε_t, ε ~ N(0, innovation_std)
            drift_angles = np.empty(n_a)
            drift_angles[0] = rng.uniform(0, 360)
            for j in range(1, n_a):
                innovation = rng.normal(0, direction_innovation_std)
                drift_angles[j] = ar1_rho * drift_angles[j - 1] + (1 - ar1_rho) * innovation

            drift_north = drift_distances * np.cos(np.radians(drift_angles))
            drift_east = drift_distances * np.sin(np.radians(drift_angles))

            cum_north = np.empty(n_a)
            cum_east = np.empty(n_a)
            cn, ce = 0.0, 0.0
            for j in range(n_a):
                cn += drift_north[j]
                ce += drift_east[j]
                if np.sqrt(cn ** 2 + ce ** 2) > max_cumulative:
                    drift_angles[j] = rng.uniform(0, 360)
                    drift_north[j] = drift_distances[j] * np.cos(np.radians(drift_angles[j]))
                    drift_east[j] = drift_distances[j] * np.sin(np.radians(drift_angles[j]))
                    cn, ce = drift_north[j], drift_east[j]
                cum_north[j] = cn
                cum_east[j] = ce

            if cartesian:
                lat_vals[aidx] += cum_north
                lon_vals[aidx] += cum_east
            else:
                lats = lat_vals[aidx].copy()
                lat_vals[aidx] += cum_north / _DLM
                lon_vals[aidx] += (
                    cum_east / (_DLM * np.cos(np.radians(lats)))
                )
            # v3.0: Speed co-spoofing for position-velocity consistency
            drift_mag = np.sqrt(cum_north ** 2 + cum_east ** 2)
            bias_scale = np.clip(drift_mag / max_cumulative, 0, 1)
            speed_bias = rng.uniform(speed_bias_lo, speed_bias_hi, size=n_a)
            speed_dir = np.sign(np.diff(drift_mag, prepend=0))
            speed_dir[speed_dir == 0] = 1
            speed_vals[aidx] += speed_bias * bias_scale * speed_dir

            is_attack_vals[aidx] = 1
            attack_type_vals[aidx] = attack_name
            total_attacked += n_a
        return total_attacked

    elif attack_name == "phantom_acceleration":
        # Physics-consistent accel → speed → position co-spoofing
        # on stationary/slow vehicles.
        scenario_vals = df["Scenario_Label"].values
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        heading_vals = df["heading_deg"].values
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        if not cartesian:
            from bsm_attacker.geo_constants import EARTH_RADIUS_M
            R = EARTH_RADIUS_M
        ramp_msgs, hold_msgs, taper_msgs = 1, 2, 2
        profile_len = ramp_msgs + hold_msgs + taper_msgs  # 5 msgs
        # Target Stationary_Brief only; fall back to flat Stationary
        n_brief = np.sum(scenario_vals == "Stationary_Brief")
        target_label = "Stationary_Brief" if n_brief > 0 else "Stationary"
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            # Filter to the target stationary label (phantom-launch from a stop).
            stat_mask = scenario_vals[seg_pos] == target_label
            stat_pos = seg_pos[stat_mask]
            if len(stat_pos) < profile_len:
                continue
            # Find contiguous runs within stat_pos
            breaks = np.where(np.diff(stat_pos) != 1)[0] + 1
            segments_list = np.split(stat_pos, breaks)
            valid_segs = [sg for sg in segments_list if len(sg) >= profile_len]
            n_segs_attack = max(1, int(len(valid_segs) * attack_ratio))
            for sg in valid_segs[:n_segs_attack]:
                aidx = sg[:profile_len]
                n_a = len(aidx)
                peak_accel = rng.uniform(2.5, 5.0)
                # Trapezoidal acceleration profile
                accel_profile = np.zeros(n_a)
                accel_profile[:ramp_msgs] = np.linspace(0, peak_accel, ramp_msgs)
                accel_profile[ramp_msgs:ramp_msgs + hold_msgs] = peak_accel
                accel_profile[ramp_msgs + hold_msgs:] = np.linspace(peak_accel, 0, taper_msgs)
                dt = 0.1
                # Integrate accel → speed, speed → displacement
                speed_profile = np.cumsum(accel_profile) * dt
                cum_disp = np.cumsum(speed_profile) * dt
                accel_vals[aidx] = accel_profile
                speed_vals[aidx] = speed_profile
                # Co-spoof position along heading
                heading_rad = np.radians(heading_vals[aidx[0]])
                off_n = cum_disp * np.cos(heading_rad)
                off_e = cum_disp * np.sin(heading_rad)
                off_n[0] = 0.0
                off_e[0] = 0.0
                if cartesian:
                    lat_vals[aidx] += off_n
                    lon_vals[aidx] += off_e
                else:
                    lats = lat_vals[aidx].copy()
                    lat_vals[aidx] += off_n / R * (180 / np.pi)
                    lon_vals[aidx] += (
                        off_e / R * (180 / np.pi) / np.cos(lats * np.pi / 180)
                    )
                is_attack_vals[aidx] = 1
                attack_type_vals[aidx] = attack_name
                total_attacked += n_a
        return total_attacked

    elif attack_name == "heading_lock":
        # Lock heading at turn-entry value, co-spoof yaw + position.
        scenario_vals = df["Scenario_Label"].values
        heading_vals = _writable("heading_deg")
        speed_vals = df["speed_mps"].values if "speed_mps" in df.columns else df["Speed_mps"].values
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        has_yaw = "yaw_rate_degs" in df.columns
        if has_yaw:
            yaw_vals = _writable("yaw_rate_degs")
        if not cartesian:
            from bsm_attacker.geo_constants import EARTH_RADIUS_M
            R = EARTH_RADIUS_M
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            # Heading-lock targets the turning phase.
            turn_mask = scenario_vals[seg_pos] == "Turning"
            turn_pos = seg_pos[turn_mask]
            if len(turn_pos) == 0:
                continue
            # Find contiguous turning segments
            breaks = np.where(np.diff(turn_pos) != 1)[0] + 1
            segments_list = np.split(turn_pos, breaks)
            for sg in segments_list:
                n_a = max(1, int(len(sg) * attack_ratio))
                aidx = sg[:n_a]
                if len(aidx) == 0:
                    continue
                locked_heading = heading_vals[aidx[0]]
                locked_heading_rad = np.radians(locked_heading)
                heading_vals[aidx] = locked_heading
                if has_yaw:
                    yaw_vals[aidx] = rng.normal(0, 0.5, size=len(aidx))
                # Co-spoof position: project along locked heading from base
                dt = 0.1
                base_lat = lat_vals[aidx[0]]
                base_lon = lon_vals[aidx[0]]
                speeds = speed_vals[aidx]
                cum_disp = np.cumsum(speeds) * dt
                cum_disp = np.insert(cum_disp[:-1], 0, 0.0)  # shift: msg 0 has no offset
                off_n = cum_disp * np.cos(locked_heading_rad)
                off_e = cum_disp * np.sin(locked_heading_rad)
                if cartesian:
                    lat_vals[aidx] = base_lat + off_n
                    lon_vals[aidx] = base_lon + off_e
                else:
                    lat_vals[aidx] = base_lat + off_n / R * (180 / np.pi)
                    lon_vals[aidx] = base_lon + (
                        off_e / R * (180 / np.pi) / np.cos(base_lat * np.pi / 180)
                    )
                is_attack_vals[aidx] = 1
                attack_type_vals[aidx] = attack_name
                total_attacked += len(aidx)
        return total_attacked

    elif attack_name == "eventual_stop":
        # Attacks Cruising rows: gradual speed reduction to zero + hold.
        scenario_vals = df["Scenario_Label"].values
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            cruise_mask = scenario_vals[seg_pos] == "Cruising"
            cruise_pos = seg_pos[cruise_mask]
            if len(cruise_pos) == 0:
                continue

            # Find contiguous cruising runs within this segment
            if len(cruise_pos) <= 1:
                runs = [cruise_pos]
            else:
                gaps = np.where(np.diff(cruise_pos) > 1)[0] + 1
                runs = np.array_split(cruise_pos, gaps)

            target_attack_msgs = int(len(cruise_pos) * attack_ratio)
            attacked_so_far = 0

            for run_pos in runs:
                if attacked_so_far >= target_attack_msgs:
                    break
                stop_dur = rng.randint(5, 16)
                hold = 5
                total_len = stop_dur + hold
                if len(run_pos) < total_len:
                    continue

                max_start = len(run_pos) - total_len
                start_p = rng.randint(0, max_start + 1) if max_start > 0 else 0
                aidx = run_pos[start_p:start_p + total_len]

                initial_speed = speed_vals[aidx[0]]
                if pd.isna(initial_speed) or initial_speed < 1.0:
                    continue

                decel = rng.uniform(-4.0, -2.0)
                dt = 0.1

                decel_idx = aidx[:stop_dur]
                elapsed = np.arange(len(decel_idx), dtype=np.float64) * dt
                decel_speeds = np.maximum(0, initial_speed + decel * elapsed)
                decel_accels = np.where(decel_speeds > 0, decel, 0.0)
                speed_vals[decel_idx] = decel_speeds
                accel_vals[decel_idx] = decel_accels

                hold_idx = aidx[stop_dur:]
                speed_vals[hold_idx] = 0.0
                accel_vals[hold_idx] = 0.0

                is_attack_vals[aidx] = 1
                attack_type_vals[aidx] = attack_name
                attacked_so_far += len(aidx)
                total_attacked += len(aidx)

        return total_attacked

    elif attack_name == "false_yield_at_intersection":
        # Ramp real speed → ~1.0 m/s over 6 BSMs while approaching an
        # intersection; co-spoof sustained deceleration.
        # Approach gate (when available): dist_to_intersection_m ≤ 100 AND
        # heading_diff_deg ≤ 45 AND speed ≥ 2.
        # Fallback: Scenario_Label == Cruising + speed ≥ 2.
        scenario_vals = df["Scenario_Label"].values
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        has_osm = ("dist_to_intersection_m" in df.columns
                   and "heading_diff_deg" in df.columns)
        if has_osm:
            d = df["dist_to_intersection_m"].fillna(np.inf).values
            h = df["heading_diff_deg"].fillna(np.inf).values
            eligible_global = (scenario_vals == "Cruising") & (
                df["speed_mps"].values >= 2.0
            ) & (d <= 100.0) & (h <= 45.0)
        else:
            eligible_global = (scenario_vals == "Cruising") & (
                df["speed_mps"].values >= 2.0
            )
        ramp_msgs = 6
        target_speed = 1.0
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            elig_mask = eligible_global[seg_pos]
            elig_pos = seg_pos[elig_mask]
            if len(elig_pos) < ramp_msgs:
                continue
            # First contiguous run of ≥ ramp_msgs in eligible positions
            breaks = np.where(np.diff(elig_pos) != 1)[0] + 1
            runs = np.split(elig_pos, breaks)
            valid = [r for r in runs if len(r) >= ramp_msgs]
            if not valid:
                continue
            win = valid[0][:ramp_msgs]
            start_v = float(df["speed_mps"].values[win[0]])
            ramp = np.linspace(start_v, target_speed, len(win))
            decel = rng.uniform(-3.0, -2.0)
            speed_vals[win] = ramp
            accel_vals[win] = decel
            is_attack_vals[win] = 1
            attack_type_vals[win] = attack_name
            total_attacked += len(win)
        return total_attacked

    elif attack_name == "speed_limit_violation":
        # Speed-compliance attack — broadcast inflated speed (1.5-2.0×
        # posted maxspeed) while Cruising. Requires a speed_limit_kmh column
        # (produces 0 rows when that road attribute is unavailable).
        scenario_vals = df["Scenario_Label"].values
        speed_vals = _writable("speed_mps")
        accel_vals = _writable("accel_long_mps2")
        is_attack_vals = _is_attack_arr
        attack_type_vals = _attack_type_arr
        if "speed_limit_kmh" not in df.columns:
            return 0
        limit_mps = (df["speed_limit_kmh"].values / 3.6)
        has_limit = ~np.isnan(limit_mps)
        real_speed = df["speed_mps"].values
        eligible_global = (
            (scenario_vals == "Cruising")
            & has_limit
            & (real_speed >= 3.0)
            & (real_speed < limit_mps * 1.2)
        )
        total_attacked = 0
        for i in range(n_segs):
            s, e = seg_starts[i], seg_ends[i]
            seg_pos = attack_pos[s:e]
            elig_pos = seg_pos[eligible_global[seg_pos]]
            if len(elig_pos) == 0:
                continue
            mult = rng.uniform(1.5, 2.0, size=len(elig_pos))
            spoof_speed = np.clip(limit_mps[elig_pos] * mult, None, 60.0)
            spoof_accel = rng.uniform(0.0, 0.5, size=len(elig_pos))
            speed_vals[elig_pos] = spoof_speed
            accel_vals[elig_pos] = spoof_accel
            is_attack_vals[elig_pos] = 1
            attack_type_vals[elig_pos] = attack_name
            total_attacked += len(elig_pos)
        return total_attacked

    else:
        raise ValueError(
            f"Unknown simple attack '{attack_name}' in vectorized path. "
            f"Add it to ROW_EXPANDING_ATTACKS if it produces extra rows."
        )


def inject_single_attack_sweep_vectorized(
    df_benign: pd.DataFrame,
    attack_name: str,
    attack_params: dict,
    cfg: dict,
) -> pd.DataFrame:
    """Vectorized attack injection.

    Drop-in replacement for ``inject_single_attack_sweep`` that is 10-50x
    faster for the 20 "simple" attacks (same row count in/out).  For the
    4 row-expanding attacks (data_replay, ghost_vehicle) it falls back to
    an optimised per-segment loop.

    Signature and output are identical to ``inject_single_attack_sweep``.
    """
    t0 = time.perf_counter()

    seed = cfg["random_seed"]
    benign_fraction = cfg["benign_fraction"]
    segment_size = cfg["segment_size"]

    df = df_benign.copy()
    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    # Assign segments: 50% benign, 50% this attack
    rng = np.random.RandomState(seed)
    slots = assign_segments(df, [attack_name], benign_fraction, segment_size, rng)

    slot_counts = pd.Series(slots).value_counts().sort_index()
    n_benign = slot_counts.get(0, 0)
    n_attack_assigned = slot_counts.get(1, 0)
    print(f"    Segments: benign={n_benign:,}, {attack_name}={n_attack_assigned:,}")

    # Start with all-benign labels
    df["is_attack"] = 0
    df["attack_type"] = "benign"

    # Extract attack_ratio (remove 'family' key)
    params = {k: v for k, v in attack_params.items() if k != "family"}

    # ------------------------------------------------------------------
    # Fast path: simple attacks (no row expansion)
    # Build seg_map inline from slots array using fast numpy boundary
    # detection — avoids the slow get_segment_boundaries() call.
    # ------------------------------------------------------------------
    if attack_name not in ROW_EXPANDING_ATTACKS:
        attack_ratio = params.get("attack_ratio", 0.15)
        cartesian = cfg.get("corpus", {}).get("coordinate_system") == "cartesian"

        # Build seg_ids / seg_starts / seg_ends from slots array
        # using fast numpy boundary detection (no DataFrame ops).
        attack_positions = np.where(slots == 1)[0]
        atk_devices = df["device_id"].values[attack_positions]

        if len(attack_positions) > 1:
            dev_breaks = atk_devices[:-1] != atk_devices[1:]
            idx_gaps = np.diff(attack_positions) > 1
            break_points = np.where(dev_breaks | idx_gaps)[0] + 1
            seg_ids = np.zeros(len(attack_positions), dtype=np.int32)
            seg_ids[break_points] = 1
            seg_ids = np.cumsum(seg_ids)
        else:
            seg_ids = np.zeros(len(attack_positions), dtype=np.int32)

        n_segs_val = seg_ids[-1] + 1 if len(seg_ids) > 0 else 0
        seg_starts_arr = np.searchsorted(seg_ids, np.arange(n_segs_val), side='left')
        seg_ends_arr = np.searchsorted(seg_ids, np.arange(n_segs_val), side='right')

        n_attacked = _vectorized_inject_simple(
            df, attack_name, attack_ratio,
            attack_positions, seg_ids, seg_starts_arr, seg_ends_arr, n_segs_val,
            seed, cartesian=cartesian,
        )
        # Apply sensor noise. noise_apply_mode in {attacked_only, symmetric}.
        # The real_site dataset_class skips noise (real BSMs already noisy).
        apply_noise = cfg.get("apply_sensor_noise", False)
        _dataset_class = cfg.get("_dataset_class", "synthetic")
        _apply_to_real = cfg.get("noise_apply_to_real_sites", False)
        _skip_real = (_dataset_class == "real_site") and (not _apply_to_real)
        if apply_noise and "is_attack" in df.columns and not _skip_real:
            from bsm_attacker.noise_model import apply_sensor_noise as _apply_noise
            _apply_mode = cfg.get("noise_apply_mode", "attacked_only")
            if _apply_mode == "symmetric":
                attack_mask = pd.Series(True, index=df.index)
            else:
                attack_mask = df["is_attack"] == 1
            _noise_mode = cfg.get("noise_mode", "realistic")
            df = _apply_noise(df, attack_mask, seed=seed, cartesian=cartesian, noise_mode=_noise_mode)
            print(f"    Sensor noise applied to {attack_mask.sum():,} rows "
                  f"(mode={_noise_mode}, apply={_apply_mode}, class={_dataset_class})")
        elif apply_noise and _skip_real:
            print(f"    Sensor noise SKIPPED (class=real_site, "
                  f"noise_apply_to_real_sites=false)")

        elapsed = time.perf_counter() - t0
        print(f"    Injected (vectorized): {n_attacked:,} attacked rows "
              f"({n_attacked / len(df) * 100:.1f}%) in {elapsed:.1f}s")
        return df

    # ------------------------------------------------------------------
    # Vectorized path for row-expanding attacks (ghost_vehicle, data_replay)
    # ------------------------------------------------------------------
    cartesian = cfg.get("corpus", {}).get("coordinate_system") == "cartesian"
    from bsm_attacker.geo_constants import DEG_LAT_TO_M, deg_lon_to_m

    # Build seg_map for row-expanding attacks
    seg_map = get_segment_boundaries(df, slots, [attack_name])
    attack_segments = list(seg_map[attack_name])
    n_segments = len(attack_segments)

    if attack_name == "ghost_vehicle":
        # Clone WHOLE device trajectories (matches the reference direct path
        # in bsm_attacker/ghost_vehicle.py:94-127). Cloning only attack-segment
        # indices creates segment-boundary derived-feature artifacts; cloning
        # full trajectories avoids them. Also skip degree-conversion on
        # Cartesian corpora (positions_are_metres heuristic).
        from bsm_attacker.geo_constants import positions_are_metres
        rng_gv = np.random.RandomState(seed + 1000)
        position_shift_range = (10, 50)
        # speed_jitter set to 0.0 for attacker-OBU realism. A 5% multiplicative
        # wobble created a kinematic fingerprint detectors locked onto; an
        # adversary in full control of transmitted fields would not self-jitter.
        speed_jitter = 0.0

        lat_col = "latitude"
        lon_col = "longitude"
        if positions_are_metres(df[lat_col].to_numpy()):
            meters_per_deg_lat = 1.0
            meters_per_deg_lon = 1.0
        else:
            meters_per_deg_lat = DEG_LAT_TO_M
            meters_per_deg_lon = deg_lon_to_m(df[lat_col].median())

        # Source vehicles = devices that had any attack segment under
        # per-vehicle scenario gating. Each contributes its FULL trajectory
        # as the ghost (same semantics as bsm_attacker/ghost_vehicle.py).
        seg_devices = np.concatenate([df.loc[idx, "device_id"].values
                                       for _, idx in attack_segments])
        unique_devs = np.unique(seg_devices)

        ghost_sequences = []
        for i, dev_id in enumerate(unique_devs):
            source_df = df[df["device_id"] == dev_id].copy()
            if len(source_df) == 0:
                continue
            ghost_df = source_df.copy()
            ghost_df["device_id"] = f"GHOST_{i:04d}"
            shift_m = rng_gv.uniform(position_shift_range[0], position_shift_range[1])
            direction = rng_gv.uniform(0, 2 * np.pi)
            delta_lat = (shift_m * np.cos(direction)) / meters_per_deg_lat
            delta_lon = (shift_m * np.sin(direction)) / meters_per_deg_lon
            ghost_df[lat_col] = ghost_df[lat_col] + delta_lat
            ghost_df[lon_col] = ghost_df[lon_col] + delta_lon
            if speed_jitter > 0 and "speed_mps" in ghost_df.columns:
                jitter = rng_gv.uniform(1 - speed_jitter, 1 + speed_jitter,
                                          size=len(ghost_df))
                ghost_df["speed_mps"] = (ghost_df["speed_mps"] * jitter).clip(lower=0)
            ghost_df["is_attack"] = 1
            ghost_df["attack_type"] = attack_name
            ghost_sequences.append(ghost_df)

        if ghost_sequences:
            attack_df = pd.concat(ghost_sequences, ignore_index=True)
            max_idx = df.index.max()
            attack_df.index = range(max_idx + 1, max_idx + 1 + len(attack_df))
            df = pd.concat([df, attack_df], ignore_index=False)
            n_attacked = len(attack_df)
        else:
            n_attacked = 0

        # Apply sensor noise (ghost_vehicle). noise_apply_mode in {attacked_only, symmetric}.
        # Refinement 1: real_site dataset_class skips noise (real BSMs already noisy).
        apply_noise = cfg.get("apply_sensor_noise", False)
        _dataset_class = cfg.get("_dataset_class", "synthetic")
        _apply_to_real = cfg.get("noise_apply_to_real_sites", False)
        _skip_real = (_dataset_class == "real_site") and (not _apply_to_real)
        if apply_noise and "is_attack" in df.columns and not _skip_real:
            from bsm_attacker.noise_model import apply_sensor_noise as _apply_noise
            _apply_mode = cfg.get("noise_apply_mode", "attacked_only")
            if _apply_mode == "symmetric":
                attack_mask = pd.Series(True, index=df.index)
            else:
                attack_mask = df["is_attack"] == 1
            _noise_mode = cfg.get("noise_mode", "realistic")
            df = _apply_noise(df, attack_mask, seed=seed, cartesian=cartesian, noise_mode=_noise_mode)
            print(f"    Sensor noise applied to {attack_mask.sum():,} rows "
                  f"(mode={_noise_mode}, apply={_apply_mode}, class={_dataset_class})")
        elif apply_noise and _skip_real:
            print(f"    Sensor noise SKIPPED (class=real_site, "
                  f"noise_apply_to_real_sites=false)")

        elapsed = time.perf_counter() - t0
        print(f"    Ghost rows appended: {n_attacked:,}")
        print(f"    Injected: {n_attacked:,} attacked rows "
              f"({n_attacked / len(df) * 100:.1f}%)")
        return df

    elif attack_name == "data_replay":
        # Vectorized data replay: copy attack segments with time delay + new IDs
        rng_dr = np.random.RandomState(seed + 2000)
        replay_delay_range = (5.0, 30.0)  # seconds

        all_attack_idx = np.concatenate([idx for _, idx in attack_segments])
        attack_df = df.loc[all_attack_idx].copy()

        # Assign replay device IDs per original device
        orig_devices = attack_df["device_id"].values
        unique_devs = np.unique(orig_devices)
        replay_id_map = {d: f"REPLAY_{i:04d}" for i, d in enumerate(unique_devs)}
        attack_df["device_id"] = [replay_id_map[d] for d in orig_devices]

        # Random time delay per replay device
        for dev_id in unique_devs:
            mask = orig_devices == dev_id
            delay_s = rng_dr.uniform(replay_delay_range[0], replay_delay_range[1])
            attack_df.loc[attack_df.index[mask], "timestamp"] += delay_s

        attack_df["is_attack"] = 1
        attack_df["attack_type"] = attack_name
        max_idx = df.index.max()
        attack_df.index = range(max_idx + 1, max_idx + 1 + len(attack_df))
        df = pd.concat([df, attack_df], ignore_index=False)
        n_attacked = len(attack_df)

        # Apply sensor noise (data_replay). noise_apply_mode in {attacked_only, symmetric}.
        # Refinement 1: real_site dataset_class skips noise (real BSMs already noisy).
        apply_noise = cfg.get("apply_sensor_noise", False)
        _dataset_class = cfg.get("_dataset_class", "synthetic")
        _apply_to_real = cfg.get("noise_apply_to_real_sites", False)
        _skip_real = (_dataset_class == "real_site") and (not _apply_to_real)
        if apply_noise and "is_attack" in df.columns and not _skip_real:
            from bsm_attacker.noise_model import apply_sensor_noise as _apply_noise
            _apply_mode = cfg.get("noise_apply_mode", "attacked_only")
            if _apply_mode == "symmetric":
                attack_mask = pd.Series(True, index=df.index)
            else:
                attack_mask = df["is_attack"] == 1
            _noise_mode = cfg.get("noise_mode", "realistic")
            df = _apply_noise(df, attack_mask, seed=seed, cartesian=cartesian, noise_mode=_noise_mode)
            print(f"    Sensor noise applied to {attack_mask.sum():,} rows "
                  f"(mode={_noise_mode}, apply={_apply_mode}, class={_dataset_class})")
        elif apply_noise and _skip_real:
            print(f"    Sensor noise SKIPPED (class=real_site, "
                  f"noise_apply_to_real_sites=false)")

        elapsed = time.perf_counter() - t0
        print(f"    Replay rows appended: {n_attacked:,}")
        print(f"    Injected: {n_attacked:,} attacked rows "
              f"({n_attacked / len(df) * 100:.1f}%)")
        return df

    # ------------------------------------------------------------------
    # Fallback: per-segment loop (should not be reached for known attacks)
    # ------------------------------------------------------------------
    df_att = df.rename(columns=CANONICAL_TO_ATTACKER)
    seg_counter = 0
    n_attacked_total = 0
    ghost_rows = []

    _shared_pipeline = AttackPipeline(random_seed=seed)
    for attacker in _shared_pipeline.attackers.values():
        attacker.verbose = False

    for device_id, seg_indices in seg_map[attack_name]:
        seg_rows = df_att.loc[seg_indices].copy()
        seg_rows["_orig_idx"] = seg_indices

        try:
            seg_attacked = inject_attack_on_segment(
                seg_rows, attack_name, params, random_seed=seed + seg_counter,
                pipeline=_shared_pipeline,
            )
        except Exception:
            seg_counter += 1
            continue

        seg_counter += 1
        n_orig = len(seg_indices)

        if len(seg_attacked) > n_orig:
            seen_idx = set()
            orig_mask = []
            for val in seg_attacked["_orig_idx"].values:
                if pd.notna(val) and int(val) not in seen_idx:
                    seen_idx.add(int(val))
                    orig_mask.append(True)
                else:
                    orig_mask.append(False)
            orig_mask = np.array(orig_mask)
            orig_part = seg_attacked.loc[orig_mask].copy()
            orig_part.index = orig_part["_orig_idx"].astype(int).values
            ghost_part_raw = seg_attacked.loc[~orig_mask].copy()

            orig_attacked = orig_part["Is_Attack"].values == 1
            orig_canonical = orig_part.rename(columns=ATTACKER_TO_CANONICAL)
            orig_attacked_idx = orig_part.index[orig_attacked]
            for col in FEATURE_COLS:
                if col in orig_canonical.columns and col in df.columns:
                    df.loc[orig_attacked_idx, col] = orig_canonical.loc[
                        orig_attacked_idx, col
                    ].values
            df.loc[orig_attacked_idx, "is_attack"] = 1
            df.loc[orig_attacked_idx, "attack_type"] = attack_name
            n_attacked_total += int(orig_attacked.sum())

            ghost_canonical = ghost_part_raw.rename(columns=ATTACKER_TO_CANONICAL)
            max_idx = df.index.max()
            ghost_canonical.index = range(
                max_idx + 1, max_idx + 1 + len(ghost_canonical)
            )
            ghost_canonical["is_attack"] = 1
            ghost_canonical["attack_type"] = attack_name
            ghost_canonical["device_id"] = f"{device_id}_ghost_{seg_counter}"
            for col in df.columns:
                if col not in ghost_canonical.columns:
                    ghost_canonical[col] = (
                        "unknown" if df[col].dtype == object else 0.0
                    )
            if "timestamp" in ghost_canonical.columns:
                ghost_canonical["timestamp"] = pd.to_numeric(
                    ghost_canonical["timestamp"], errors="coerce"
                ).fillna(0.0)
            ghost_rows.append(ghost_canonical[df.columns])
            n_attacked_total += len(ghost_canonical)
            continue

        elif len(seg_attacked) < n_orig:
            continue

        seg_attacked.index = seg_indices
        actually_attacked = seg_attacked["Is_Attack"].values == 1
        attacked_indices = seg_indices[actually_attacked]
        seg_canonical = seg_attacked.rename(columns=ATTACKER_TO_CANONICAL)
        for col in FEATURE_COLS:
            if col in seg_canonical.columns and col in df.columns:
                df.loc[attacked_indices, col] = seg_canonical.loc[
                    attacked_indices, col
                ].values
        df.loc[attacked_indices, "is_attack"] = 1
        df.loc[attacked_indices, "attack_type"] = attack_name
        n_attacked_total += int(actually_attacked.sum())

    # Append ghost rows
    if ghost_rows:
        ghost_df = pd.concat(ghost_rows, ignore_index=False)
        df = pd.concat([df, ghost_df], ignore_index=False)
        print(f"    Ghost rows appended: {len(ghost_df):,}")

    df = df.drop(columns=["_orig_idx"], errors="ignore")

    # Apply sensor noise. noise_apply_mode in {attacked_only, symmetric}.
    # Refinement 1: real_site dataset_class skips noise (real BSMs already noisy).
    apply_noise = cfg.get("apply_sensor_noise", False)
    _dataset_class = cfg.get("_dataset_class", "synthetic")
    _apply_to_real = cfg.get("noise_apply_to_real_sites", False)
    _skip_real = (_dataset_class == "real_site") and (not _apply_to_real)
    if apply_noise and "is_attack" in df.columns and not _skip_real:
        from bsm_attacker.noise_model import apply_sensor_noise
        cartesian = cfg.get("corpus", {}).get("coordinate_system") == "cartesian"
        _apply_mode = cfg.get("noise_apply_mode", "attacked_only")
        if _apply_mode == "symmetric":
            attack_mask = pd.Series(True, index=df.index)
        else:
            attack_mask = df["is_attack"] == 1
        _noise_mode = cfg.get("noise_mode", "realistic")
        df = apply_sensor_noise(df, attack_mask, seed=seed, cartesian=cartesian, noise_mode=_noise_mode)
        print(f"    Sensor noise applied to {attack_mask.sum():,} rows "
              f"(mode={_noise_mode}, apply={_apply_mode}, class={_dataset_class})")
    elif apply_noise and _skip_real:
        print(f"    Sensor noise SKIPPED (class=real_site, "
              f"noise_apply_to_real_sites=false)")

    elapsed = time.perf_counter() - t0
    print(f"    Injected: {n_attacked_total:,} attacked rows "
          f"({n_attacked_total / len(df) * 100:.1f}%) in {elapsed:.1f}s")
    return df


def inject_attack_on_segment(
    segment_df: pd.DataFrame,
    attack_name: str,
    attack_params: dict,
    random_seed: int,
    pipeline: "AttackPipeline | None" = None,
) -> pd.DataFrame:
    """Run attacker on a single segment with fresh state."""
    if pipeline is None:
        pipeline = AttackPipeline(random_seed=random_seed)
    else:
        # Reseed the specific attacker for reproducibility
        np.random.seed(random_seed)
    return pipeline.inject_single_attack(
        segment_df, attack_name, auto_preprocess=False, **attack_params,
    )


def inject_single_attack_sweep(
    df_benign: pd.DataFrame,
    attack_name: str,
    attack_params: dict,
    cfg: dict,
) -> pd.DataFrame:
    """Inject one attack type using episode-correct segment partitioning.

    Returns DataFrame with is_attack and attack_type columns.
    """
    seed = cfg["random_seed"]
    benign_fraction = cfg["benign_fraction"]
    segment_size = cfg["segment_size"]

    df = df_benign.copy()
    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    # Assign segments: 50% benign, 50% this attack
    rng = np.random.RandomState(seed)
    slots = assign_segments(df, [attack_name], benign_fraction, segment_size, rng)

    slot_counts = pd.Series(slots).value_counts().sort_index()
    n_benign = slot_counts.get(0, 0)
    n_attack_assigned = slot_counts.get(1, 0)
    print(f"    Segments: benign={n_benign:,}, {attack_name}={n_attack_assigned:,}")

    seg_map = get_segment_boundaries(df, slots, [attack_name])

    # Start with all-benign labels
    df["is_attack"] = 0
    df["attack_type"] = "benign"

    # Extract only the attack_ratio (remove 'family' key before passing to attacker)
    params = {k: v for k, v in attack_params.items() if k != "family"}

    # --- OPT: Rename columns ONCE, not per-segment ---
    df_att = df.rename(columns=CANONICAL_TO_ATTACKER)

    n_segments = len(seg_map[attack_name])
    n_workers = int(os.environ.get("INJECTION_WORKERS", "1"))
    if n_workers <= 0:
        n_workers = max(1, os.cpu_count() // 2)

    # --- Parallel injection path (CPU-only, no CUDA in workers) ---
    if n_workers > 1 and n_segments > n_workers * 2:
        import multiprocessing as _mp

        print(f"    Injecting {n_segments} segments across {n_workers} CPU cores...")

        seg_list = list(seg_map[attack_name])
        batch_size = max(1, len(seg_list) // (n_workers * 4))
        batches = [seg_list[i:i + batch_size]
                   for i in range(0, len(seg_list), batch_size)]

        # Use fork context (shares memory, avoids pickle of large df)
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            worker_args = [
                (batch, attack_name, params, seed + i * batch_size, df_att)
                for i, batch in enumerate(batches)
            ]
            batch_results = pool.map(_injection_worker, worker_args)

        # Collect results
        seg_counter = 0
        n_attacked_total = 0
        ghost_rows = []

        for batch in batch_results:
            if batch is None:
                continue
            for item in batch:
                if item is None:
                    seg_counter += 1
                    continue
                dev_id, seg_indices, seg_attacked, _ = item
                seg_counter += 1
                n_orig = len(seg_indices)

                if len(seg_attacked) > n_orig:
                    # Ghost/replay rows — same logic as sequential path
                    _orig_vals = seg_attacked["_orig_idx"].values
                    _seen = set()
                    _om = []
                    for v in _orig_vals:
                        if pd.notna(v) and int(v) not in _seen:
                            _seen.add(int(v))
                            _om.append(True)
                        else:
                            _om.append(False)
                    orig_mask = np.array(_om)
                    orig_part = seg_attacked.loc[orig_mask].copy()
                    orig_part.index = orig_part["_orig_idx"].astype(int).values
                    ghost_part_raw = seg_attacked.loc[~orig_mask].copy()

                    orig_attacked = orig_part["Is_Attack"].values == 1
                    orig_canonical = orig_part.rename(columns=ATTACKER_TO_CANONICAL)
                    orig_attacked_idx = orig_part.index[orig_attacked]
                    for col in FEATURE_COLS:
                        if col in orig_canonical.columns and col in df.columns:
                            df.loc[orig_attacked_idx, col] = orig_canonical.loc[orig_attacked_idx, col].values
                    df.loc[orig_attacked_idx, "is_attack"] = 1
                    df.loc[orig_attacked_idx, "attack_type"] = attack_name
                    n_attacked_total += int(orig_attacked.sum())

                    ghost_canonical = ghost_part_raw.rename(columns=ATTACKER_TO_CANONICAL)
                    max_idx = df.index.max()
                    ghost_canonical.index = range(max_idx + 1, max_idx + 1 + len(ghost_canonical))
                    ghost_canonical["is_attack"] = 1
                    ghost_canonical["attack_type"] = attack_name
                    ghost_canonical["device_id"] = f"{dev_id}_ghost_{seg_counter}"
                    for col in df.columns:
                        if col not in ghost_canonical.columns:
                            ghost_canonical[col] = "unknown" if df[col].dtype == object else 0.0
                    if "timestamp" in ghost_canonical.columns:
                        ghost_canonical["timestamp"] = pd.to_numeric(
                            ghost_canonical["timestamp"], errors="coerce"
                        ).fillna(0.0)
                    ghost_rows.append(ghost_canonical[df.columns])
                    n_attacked_total += len(ghost_canonical)

                elif len(seg_attacked) < n_orig:
                    continue

                else:
                    seg_attacked.index = seg_indices
                    actually_attacked = seg_attacked["Is_Attack"].values == 1
                    attacked_indices = seg_indices[actually_attacked]
                    seg_canonical = seg_attacked.rename(columns=ATTACKER_TO_CANONICAL)
                    for col in FEATURE_COLS:
                        if col in seg_canonical.columns and col in df.columns:
                            df.loc[attacked_indices, col] = seg_canonical.loc[attacked_indices, col].values
                    df.loc[attacked_indices, "is_attack"] = 1
                    df.loc[attacked_indices, "attack_type"] = attack_name
                    n_attacked_total += int(actually_attacked.sum())

    else:
        # --- Sequential path (original behavior) ---
        seg_counter = 0
        n_attacked_total = 0
        ghost_rows = []

        _shared_pipeline = AttackPipeline(random_seed=seed)
        for attacker in _shared_pipeline.attackers.values():
            attacker.verbose = False

        for device_id, seg_indices in seg_map[attack_name]:
            seg_rows = df_att.loc[seg_indices].copy()
            seg_rows["_orig_idx"] = seg_indices

            try:
                seg_attacked = inject_attack_on_segment(
                    seg_rows, attack_name, params, random_seed=seed + seg_counter,
                    pipeline=_shared_pipeline,
                )
            except Exception as e:
                seg_counter += 1
                continue

            seg_counter += 1
            n_orig = len(seg_indices)

            if len(seg_attacked) > n_orig:
                seen_idx = set()
                orig_mask = []
                for val in seg_attacked["_orig_idx"].values:
                    if pd.notna(val) and int(val) not in seen_idx:
                        seen_idx.add(int(val))
                        orig_mask.append(True)
                    else:
                        orig_mask.append(False)
                orig_mask = np.array(orig_mask)
                orig_part = seg_attacked.loc[orig_mask].copy()
                orig_part.index = orig_part["_orig_idx"].astype(int).values
                ghost_part_raw = seg_attacked.loc[~orig_mask].copy()

                orig_attacked = orig_part["Is_Attack"].values == 1
                orig_canonical = orig_part.rename(columns=ATTACKER_TO_CANONICAL)
                orig_attacked_idx = orig_part.index[orig_attacked]
                for col in FEATURE_COLS:
                    if col in orig_canonical.columns and col in df.columns:
                        df.loc[orig_attacked_idx, col] = orig_canonical.loc[orig_attacked_idx, col].values
                df.loc[orig_attacked_idx, "is_attack"] = 1
                df.loc[orig_attacked_idx, "attack_type"] = attack_name
                n_attacked_total += int(orig_attacked.sum())

                ghost_canonical = ghost_part_raw.rename(columns=ATTACKER_TO_CANONICAL)
                max_idx = df.index.max()
                ghost_canonical.index = range(max_idx + 1, max_idx + 1 + len(ghost_canonical))
                ghost_canonical["is_attack"] = 1
                ghost_canonical["attack_type"] = attack_name
                ghost_canonical["device_id"] = f"{device_id}_ghost_{seg_counter}"
                for col in df.columns:
                    if col not in ghost_canonical.columns:
                        ghost_canonical[col] = "unknown" if df[col].dtype == object else 0.0
                if "timestamp" in ghost_canonical.columns:
                    ghost_canonical["timestamp"] = pd.to_numeric(
                        ghost_canonical["timestamp"], errors="coerce"
                    ).fillna(0.0)
                ghost_rows.append(ghost_canonical[df.columns])
                n_attacked_total += len(ghost_canonical)
                continue

            elif len(seg_attacked) < n_orig:
                continue

            seg_attacked.index = seg_indices
            actually_attacked = seg_attacked["Is_Attack"].values == 1
            attacked_indices = seg_indices[actually_attacked]
            seg_canonical = seg_attacked.rename(columns=ATTACKER_TO_CANONICAL)
            for col in FEATURE_COLS:
                if col in seg_canonical.columns and col in df.columns:
                    df.loc[attacked_indices, col] = seg_canonical.loc[attacked_indices, col].values
            df.loc[attacked_indices, "is_attack"] = 1
            df.loc[attacked_indices, "attack_type"] = attack_name
            n_attacked_total += int(actually_attacked.sum())

    # Append ghost rows from row-expanding attacks
    if ghost_rows:
        ghost_df = pd.concat(ghost_rows, ignore_index=False)
        df = pd.concat([df, ghost_df], ignore_index=False)
        print(f"    Ghost rows appended: {len(ghost_df):,}")

    # Clean up
    df = df.drop(columns=["_orig_idx"], errors="ignore")

    print(f"    Injected: {n_attacked_total:,} attacked rows "
          f"({n_attacked_total / len(df) * 100:.1f}%)")

    return df


# ======================================================================
# Real-site data cleaning
# ======================================================================

def strip_parked(
    df: pd.DataFrame, min_duration_s: float = 60.0, speed_thresh: float = 0.5,
) -> pd.DataFrame:
    """Remove continuous stationary segments longer than min_duration_s.

    Keeps brief stops (traffic lights, stop signs). Only removes segments
    where a device is stationary for >60s (parked, depot, warmup).
    """
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_s"
    df = df.sort_values(["device_id", ts_col]).reset_index(drop=True)
    is_stat = (df["speed_mps"] < speed_thresh).astype(int)
    dev_change = df["device_id"].ne(df["device_id"].shift())
    grp_id = (is_stat.ne(is_stat.shift()) | dev_change).cumsum()

    seg_info = df.groupby(grp_id).agg(
        ts_min=(ts_col, "first"),
        ts_max=(ts_col, "last"),
        is_stat=("speed_mps", lambda x: (x < speed_thresh).all()),
        count=("speed_mps", "size"),
    )
    seg_info["duration"] = seg_info["ts_max"] - seg_info["ts_min"]
    parked_segs = seg_info[
        seg_info["is_stat"] & (seg_info["duration"] > min_duration_s)
    ].index
    drop_mask = grp_id.isin(parked_segs)

    n_dropped = drop_mask.sum()
    if n_dropped > 0:
        print(f"    Parked-strip: removed {n_dropped:,} rows ({n_dropped/len(df)*100:.1f}%) "
              f"from {len(parked_segs)} segments >{min_duration_s:.0f}s")
    return df[~drop_mask].reset_index(drop=True)


def downsample_adaptive(
    df: pd.DataFrame, target_hz: float = 1.0,
) -> pd.DataFrame:
    """Downsample to ~target_hz based on measured median Δt.

    Handles variable-rate ETSI CAM data by computing the actual rate
    and taking every Nth row to approximate target_hz.
    """
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_s"
    df = df.sort_values(["device_id", ts_col]).reset_index(drop=True)
    dt = df.groupby("device_id")[ts_col].diff().dropna()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return df
    median_dt = dt.median()
    actual_hz = 1.0 / median_dt
    if actual_hz <= target_hz * 1.5:
        return df  # already at or below target rate
    step = max(1, int(round(actual_hz / target_hz)))
    mask = df.groupby("device_id").cumcount() % step == 0
    result = df[mask].reset_index(drop=True)
    print(f"    Downsample: {actual_hz:.1f} Hz -> ~{target_hz} Hz "
          f"(every {step}th row, {len(df):,} -> {len(result):,})")
    return result


def clean_real_site(df: pd.DataFrame, site_name: str = "") -> pd.DataFrame:
    """Full cleaning pipeline for real-site data: parked-strip → 1 Hz downsample.

    Should be called BEFORE label_scenarios() and inject_attack().
    Does NOT apply to VeReMi (already 1 Hz, minimal parked).
    """
    n_orig = len(df)
    n_dev = df["device_id"].nunique()
    print(f"  Cleaning {site_name}: {n_orig:,} rows, {n_dev} devices")
    df = strip_parked(df)
    df = downsample_adaptive(df, target_hz=1.0)
    df = strip_parked(df)  # second pass after downsample (merging can create new parked segments)
    print(f"    Final: {len(df):,} rows ({len(df)/n_orig*100:.1f}% retained)")
    return df


# ======================================================================
# Feature engineering
# ======================================================================

def add_derived_features(
    df: pd.DataFrame, cartesian: bool = False, max_gap_s: float = 5.0,
) -> pd.DataFrame:
    """Compute per-device temporal deltas and physics consistency features.

    Args:
        cartesian: If True, positions are already in metres (SUMO XY) and
                   no deg→m conversion is applied. Used for VeReMi Extension.
        max_gap_s: Drop rows where Δt to previous row exceeds this threshold.
                   Prevents spurious deltas across inter-session gaps or sparse
                   public datasets (Tampa, Wyoming). Set to 0 to disable.
    """
    df = df.copy()
    df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    grp = df.groupby("device_id")

    # Use NaN for first row per device (not 0.0) to avoid spurious zeros.
    # First rows are dropped at the end of this function.
    _first_row_mask = grp.cumcount() == 0

    # Gap-filter: drop rows where Δt exceeds max_gap_s
    dt_raw = grp["timestamp"].diff()
    _gap_mask = (dt_raw > max_gap_s) if max_gap_s > 0 else pd.Series(False, index=df.index)
    _drop_mask = _first_row_mask | _gap_mask
    n_gaps = _gap_mask.sum()
    if n_gaps > 0:
        print(f"    Gap-filter: dropped {n_gaps:,} rows with dt > {max_gap_s}s")

    df["speed_delta"] = grp["speed_mps"].diff()
    df["heading_delta"] = grp["heading_deg"].diff()
    # Wrap heading delta to [-180, 180]
    df["heading_delta"] = (df["heading_delta"] + 180) % 360 - 180

    df["accel_jerk"] = grp["accel_long_mps2"].diff()

    if cartesian:
        # Positions already in metres (SUMO Cartesian)
        df["lat_delta"] = grp["latitude"].diff()
        df["lon_delta"] = grp["longitude"].diff()
    else:
        # GPS degrees → metres
        from bsm_attacker.geo_constants import DEG_LAT_TO_M
        df["lat_delta"] = grp["latitude"].diff() * DEG_LAT_TO_M
        df["lon_delta"] = (
            grp["longitude"].diff()
            * DEG_LAT_TO_M
            * np.cos(np.radians(df["latitude"]))
        )  # deg -> m, corrected for latitude

    dt = grp["timestamp"].diff()
    dt = dt.clip(lower=0.01)
    expected_dist = df["speed_mps"] * dt

    if cartesian:
        actual_dist = np.sqrt(
            grp["latitude"].diff() ** 2
            + grp["longitude"].diff() ** 2
        )
    else:
        # Use the already-corrected lat_delta / lon_delta (in metres)
        actual_dist = np.sqrt(
            df["lat_delta"] ** 2 + df["lon_delta"] ** 2
        )

    df["pos_vel_consistency"] = (
        actual_dist / expected_dist.clip(lower=0.01)
    ).clip(upper=10.0)

    # Drop first row per device + gap rows (all diff-based features are invalid)
    df = df.loc[~_drop_mask].reset_index(drop=True)

    return df


# ======================================================================
# Dataset & Model
# ======================================================================

class BSMSequenceDataset(Dataset):
    """Sliding-window dataset respecting per-device boundaries.

    Vectorized: computes device boundaries once, then builds all valid
    window indices in a single numpy operation. ~20x faster than the
    per-device Python loop for large datasets (e.g., SPMD 20M rows).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 device_ids: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        n_feat = features.shape[1]

        # Find device boundaries using sorted order
        # (data should already be sorted by device_id, but ensure it)
        sort_idx = np.lexsort((np.arange(len(device_ids)), device_ids))
        sorted_devs = device_ids[sort_idx]
        sorted_feat = features[sort_idx]
        sorted_lab = labels[sort_idx]

        # Find where device changes (boundaries)
        changes = np.where(sorted_devs[1:] != sorted_devs[:-1])[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(sorted_devs)]])
        lengths = ends - starts

        # For each device, compute valid window start indices
        # A device with L rows produces L - seq_len + 1 windows
        valid_lengths = np.maximum(lengths - seq_len + 1, 0)
        total_windows = valid_lengths.sum()

        if total_windows == 0:
            raise ValueError(f"Not enough data for seq_len={seq_len}")

        # Build flat index array of all valid window start positions
        # For device i starting at starts[i] with valid_lengths[i] windows:
        # indices are starts[i], starts[i]+1, ..., starts[i]+valid_lengths[i]-1
        window_starts = np.empty(total_windows, dtype=np.int64)
        pos = 0
        for i in range(len(starts)):
            vl = valid_lengths[i]
            if vl > 0:
                window_starts[pos:pos + vl] = np.arange(starts[i], starts[i] + vl)
                pos += vl

        # Build all windows using advanced indexing
        # Each window: [start, start+1, ..., start+seq_len-1]
        offsets = np.arange(seq_len)
        all_indices = window_starts[:, None] + offsets[None, :]  # (total_windows, seq_len)

        self.X = sorted_feat[all_indices]  # (total_windows, seq_len, n_feat)
        lbl_windows = sorted_lab[all_indices]  # (total_windows, seq_len)
        self.y = (lbl_windows.sum(axis=1) > 0).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_labels, all_scores = [], []

    is_sklearn = getattr(model, "is_sklearn", False)

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)

        if is_sklearn:
            scores = logits.detach().cpu().numpy()
            all_scores.extend(scores)
        else:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.extend(probs)

        all_labels.extend(y_batch.detach().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_scores)

    # Threshold: use model's calibrated threshold if available,
    # otherwise compute optimal F1 threshold for unsupervised models
    is_unsup = getattr(model, "is_unsupervised", False)
    stored_thresh = getattr(model, "threshold_", None)
    if stored_thresh is not None:
        threshold = stored_thresh
    elif is_unsup and len(np.unique(y_true)) > 1:
        from sklearn.metrics import precision_recall_curve as _prc
        _p, _r, _t = _prc(y_true, y_prob)
        _f1 = 2 * _p[:-1] * _r[:-1] / (_p[:-1] + _r[:-1] + 1e-8)
        threshold = float(_t[np.argmax(_f1)]) if len(_f1) > 0 else 0.0
    elif is_sklearn and is_unsup:
        threshold = 0.0
    else:
        threshold = 0.5
    y_pred = (y_prob > threshold).astype(float)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }

    # FPR + AUROC + AUPRC + f1_optimal + fpr_at_95recall
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))

        # F1 at optimal threshold
        # Note: precision_recall_curve appends a sentinel (precision=1, recall=0)
        # at the end, giving f1=0 for that point. Harmless for max(), but if
        # recovering the optimal threshold, use thresholds[:-1] (one shorter).
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        metrics["f1_optimal"] = float(f1s.max())

        # FPR at 95% recall
        fprs_roc, tprs_roc, _ = roc_curve(y_true, y_prob)
        idx_95 = np.searchsorted(tprs_roc, 0.95)
        metrics["fpr_at_95recall"] = float(fprs_roc[min(idx_95, len(fprs_roc) - 1)])
    else:
        metrics["fpr"] = 0.0
        metrics["auc"] = 0.0
        metrics["auprc"] = 0.0
        metrics["f1_optimal"] = 0.0
        metrics["fpr_at_95recall"] = 1.0

    return metrics


def evaluate_v62(model, loader, device, save_dir=None):
    """Evaluate with predictions saving, balanced eval, and bootstrap CIs.

    Returns standard metrics dict plus:
      - balanced_auroc, balanced_auroc_std (100 subsamples)
      - auroc_ci_lower, auroc_ci_upper (1000 bootstrap resamples)
      - n_positive, n_negative (already in base evaluate)

    If save_dir is provided, saves predictions.npz with y_true, y_score.
    """
    # Run base evaluation to get scores
    model.eval()
    all_labels, all_scores_list = [], []
    is_sklearn = getattr(model, "is_sklearn", False)

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        if is_sklearn:
            scores = logits.detach().cpu().numpy()
        else:
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores_list.extend(scores)
        all_labels.extend(y_batch.detach().numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_scores_list)

    # Base metrics — use model's calibrated threshold if available,
    # otherwise use optimal F1 threshold from ROC curve
    is_unsup = getattr(model, "is_unsupervised", False)
    stored_thresh = getattr(model, "threshold_", None)
    if stored_thresh is not None:
        threshold = stored_thresh
    elif is_unsup:
        # Compute optimal F1 threshold from scores
        from sklearn.metrics import precision_recall_curve
        precs_t, recs_t, threshs_t = precision_recall_curve(y_true, y_score)
        f1s_t = 2 * precs_t[:-1] * recs_t[:-1] / (precs_t[:-1] + recs_t[:-1] + 1e-8)
        if len(f1s_t) > 0:
            threshold = float(threshs_t[np.argmax(f1s_t)])
        else:
            threshold = 0.0 if (is_sklearn and is_unsup) else 0.5
    else:
        threshold = 0.5
    y_pred = (y_score > threshold).astype(float)

    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }

    if len(np.unique(y_true)) > 1 and y_true.sum() > 0:
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))

        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        metrics["f1_optimal"] = float(f1s.max())

        fprs_roc, tprs_roc, _ = roc_curve(y_true, y_score)
        idx_95 = np.searchsorted(tprs_roc, 0.95)
        metrics["fpr_at_95recall"] = float(fprs_roc[min(idx_95, len(fprs_roc) - 1)])

        # ── Balanced evaluation (100 subsamples) ──
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos = len(pos_idx)
        n_neg_sample = min(len(neg_idx), n_pos * 10)  # 10:1 ratio

        rng = np.random.RandomState(42)
        balanced_aurocs = []
        for _ in range(100):
            neg_sub = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            bal_idx = np.concatenate([pos_idx, neg_sub])
            try:
                bal_auroc = roc_auc_score(y_true[bal_idx], y_score[bal_idx])
                balanced_aurocs.append(bal_auroc)
            except ValueError:
                pass

        if balanced_aurocs:
            metrics["balanced_auroc"] = float(np.mean(balanced_aurocs))
            metrics["balanced_auroc_std"] = float(np.std(balanced_aurocs))
        else:
            metrics["balanced_auroc"] = 0.0
            metrics["balanced_auroc_std"] = 0.0

        # ── Bootstrap CIs (1000 resamples) ──
        boot_aurocs = []
        for _ in range(1000):
            # Stratified bootstrap
            pos_boot = rng.choice(pos_idx, size=n_pos, replace=True)
            neg_boot = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([pos_boot, neg_boot])
            try:
                boot_auroc = roc_auc_score(y_true[boot_idx], y_score[boot_idx])
                boot_aurocs.append(boot_auroc)
            except ValueError:
                pass

        if boot_aurocs:
            metrics["auroc_ci_lower"] = float(np.percentile(boot_aurocs, 2.5))
            metrics["auroc_ci_upper"] = float(np.percentile(boot_aurocs, 97.5))
        else:
            metrics["auroc_ci_lower"] = 0.0
            metrics["auroc_ci_upper"] = 0.0

        # Flag
        metrics["low_n"] = n_pos < 50
        metrics["wide_ci"] = (metrics["auroc_ci_upper"] - metrics["auroc_ci_lower"]) > 0.05

    else:
        for k in ["fpr", "auc", "auprc", "f1_optimal", "fpr_at_95recall",
                   "balanced_auroc", "balanced_auroc_std",
                   "auroc_ci_lower", "auroc_ci_upper"]:
            metrics[k] = 0.0
        metrics["low_n"] = True
        metrics["wide_ci"] = True

    # ── Save predictions ──
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / "predictions.npz",
            y_true=y_true, y_score=y_score,
        )

    return metrics
