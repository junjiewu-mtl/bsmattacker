"""
NYC CV Pilot EVENT Data Parser
===============================

Parses the nested bsmList JSON from NYC Connected Vehicle Pilot EVENT data
into flat host/target BSM DataFrames with canonical column names.

Data source: NYC CV Pilot Deployment Program
File: data/nyc_cv_pilot_events.csv

Coordinate system: Cartesian (X_m, Y_m) in metres — obfuscated, NOT GPS lat/lon.
"""

import json
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Allow large JSON fields
csv_FIELD_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB


# NYC coreData → canonical column mapping
NYC_TO_CANONICAL = {
    'id': 'device_id',
    'T_s': 'timestamp',
    'X_m': 'latitude',       # Cartesian metres, NOT GPS
    'Y_m': 'longitude',      # Cartesian metres, NOT GPS
    'speed_mps': 'speed_mps',
    'heading_deg': 'heading_deg',
}

ACCEL_FIELDS = {
    'long_mpss': 'accel_long_mps2',
    'lat_mpss': 'accel_lat_mps2',
    'vert_mpss': 'accel_vert_mps2',
    'yaw_dps': 'yaw_rate_degs',
}


def flatten_bsm_core_data(bsm_entry: dict) -> dict:
    """Extract flat dict from nested bsmRecord.bsmMsg.coreData."""
    core = bsm_entry['bsmRecord']['bsmMsg']['coreData']
    row = {}
    for nyc_key, canon_key in NYC_TO_CANONICAL.items():
        row[canon_key] = core.get(nyc_key)
    accel_set = core.get('accelSet', {})
    for accel_key, canon_key in ACCEL_FIELDS.items():
        row[canon_key] = accel_set.get(accel_key, 0.0)
    return row


def parse_event_bsms(row: pd.Series) -> Tuple[Optional[pd.DataFrame],
                                                Optional[pd.DataFrame], dict]:
    """Parse one event row → (host_df, target_df, event_meta).

    Returns (None, None, meta) if bsmList parsing fails or no BSMs found.
    """
    event_meta = {
        'event_type': row.get('eventHeader_eventType', ''),
        'host_id': str(row.get('eventHeader_hostVehID', '')),
        'target_id': str(row.get('eventHeader_targetVehID', '')),
        'event_idx': row.name if hasattr(row, 'name') else None,
    }

    bsm_json = row.get('bsmList', '')
    if not bsm_json or pd.isna(bsm_json):
        return None, None, event_meta

    try:
        bsm_list = json.loads(bsm_json)
    except (json.JSONDecodeError, TypeError):
        return None, None, event_meta

    if not bsm_list:
        return None, None, event_meta

    # Flatten all BSMs
    records = []
    for entry in bsm_list:
        try:
            records.append(flatten_bsm_core_data(entry))
        except (KeyError, TypeError):
            continue

    if not records:
        return None, None, event_meta

    all_df = pd.DataFrame(records)

    # Split by vehicle ID
    host_id = event_meta['host_id']
    target_id = event_meta['target_id']

    host_df = all_df[all_df['device_id'] == host_id].copy()
    target_df = all_df[all_df['device_id'] == target_id].copy()

    # Sort by timestamp
    for df in [host_df, target_df]:
        if not df.empty:
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

    event_meta['total_bsms'] = len(all_df)
    event_meta['host_bsms'] = len(host_df)
    event_meta['target_bsms'] = len(target_df)
    event_meta['unique_ids'] = all_df['device_id'].nunique()

    return (host_df if not host_df.empty else None,
            target_df if not target_df.empty else None,
            event_meta)


def count_target_bsms_in_window(target_df: Optional[pd.DataFrame],
                                 window: Tuple[float, float] = (-5.0, 0.0)) -> int:
    """Count target BSMs with T_s in [window[0], window[1]]."""
    if target_df is None or target_df.empty:
        return 0
    mask = (target_df['timestamp'] >= window[0]) & (target_df['timestamp'] <= window[1])
    return int(mask.sum())


def load_events_by_type(csv_path: str, event_type: str,
                        min_target_bsms: int = 5,
                        pre_trigger_window: Tuple[float, float] = (-5.0, 0.0),
                        max_events: Optional[int] = None) -> list:
    """Load and parse all events of given type.

    Args:
        csv_path: Path to NYC EVENT CSV
        event_type: e.g. 'fcw', 'ima'
        min_target_bsms: Minimum target BSMs in pre-trigger window
        pre_trigger_window: (start, end) in seconds relative to trigger
        max_events: Optional cap on returned events

    Returns:
        List of (host_df, target_df, event_meta) tuples that pass filtering.
        Also prints filtering statistics.
    """
    import csv as csv_module
    csv_module.field_size_limit(csv_FIELD_SIZE_LIMIT)

    df = pd.read_csv(csv_path)

    # Filter to event type
    type_mask = df['eventHeader_eventType'].str.lower() == event_type.lower()
    event_df = df[type_mask]
    total_events = len(event_df)

    results = []
    parse_fail = 0
    no_target = 0
    filter_fail = 0

    for idx, row in event_df.iterrows():
        host_df, target_df, meta = parse_event_bsms(row)

        if host_df is None:
            parse_fail += 1
            continue

        if target_df is None:
            no_target += 1
            continue

        n_target_pre = count_target_bsms_in_window(target_df, pre_trigger_window)
        meta['target_bsms_pre_trigger'] = n_target_pre

        if n_target_pre < min_target_bsms:
            filter_fail += 1
            continue

        results.append((host_df, target_df, meta))

        if max_events and len(results) >= max_events:
            break

    print(f"\n{'='*60}")
    print(f"NYC EVENT Parser: {event_type.upper()}")
    print(f"{'='*60}")
    print(f"Total {event_type} events:          {total_events}")
    print(f"Parse failures:              {parse_fail}")
    print(f"No target BSMs:              {no_target}")
    print(f"Filter fail (< {min_target_bsms} pre-trigger): {filter_fail}")
    print(f"Events passing filter:       {len(results)}")
    pass_rate = (len(results) / total_events * 100) if total_events else 0.0
    print(f"Pass rate:                   {pass_rate:.1f}%")
    print(f"{'='*60}\n")

    return results


def check_filter_pass_rates(csv_path: str, event_type: str,
                             thresholds: list = [3, 5, 7, 10],
                             window: Tuple[float, float] = (-5.0, 0.0)):
    """Check what % of events pass at various min_target_bsm thresholds.

    Use this to decide the right filter threshold before committing.
    """
    import csv as csv_module
    csv_module.field_size_limit(csv_FIELD_SIZE_LIMIT)

    df = pd.read_csv(csv_path)
    type_mask = df['eventHeader_eventType'].str.lower() == event_type.lower()
    event_df = df[type_mask]

    counts = []
    for _, row in event_df.iterrows():
        _, target_df, _ = parse_event_bsms(row)
        n = count_target_bsms_in_window(target_df, window)
        counts.append(n)

    counts = np.array(counts)
    total = len(counts)

    print(f"\nFilter pass rates for {event_type.upper()} ({total} events):")
    print(f"  Target BSMs in T_s ∈ [{window[0]}, {window[1]}]")
    print(f"  {'Threshold':<12} {'Pass':<8} {'Rate':<8}")
    print(f"  {'-'*28}")
    for t in thresholds:
        n_pass = (counts >= t).sum()
        print(f"  ≥{t:<10} {n_pass:<8} {n_pass/total*100:.1f}%")

    # Also report distribution stats
    print(f"\n  Distribution of target BSMs in window:")
    print(f"  min={counts.min()}, p25={np.percentile(counts, 25):.0f}, "
          f"median={np.median(counts):.0f}, p75={np.percentile(counts, 75):.0f}, "
          f"max={counts.max()}")

    return counts


if __name__ == '__main__':
    import yaml

    with open('safety_replay/configs/replay.yaml') as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg['data']['nyc_event_csv']

    # Check filter pass rates for both event types
    for etype in ['fcw', 'ima']:
        check_filter_pass_rates(csv_path, etype)

    # Parse 10 FCW events as validation
    print("\n--- Validating parser on 10 FCW events ---")
    events = load_events_by_type(csv_path, 'fcw', min_target_bsms=3, max_events=10)
    if events:
        host_df, target_df, meta = events[0]
        print(f"\nFirst event meta: {meta}")
        print(f"Host columns: {list(host_df.columns)}")
        print(f"Host shape: {host_df.shape}")
        print(f"Target shape: {target_df.shape}")
        print(f"Host T_s range: [{host_df['timestamp'].min():.2f}, {host_df['timestamp'].max():.2f}]")
        print(f"Target T_s range: [{target_df['timestamp'].min():.2f}, {target_df['timestamp'].max():.2f}]")
