"""Data-integrity pass: replace ETSI/J2735 "value unavailable" sentinel codes with NaN.

Some real-world V2X logs carry the BSM/CAM reserved "unavailable" placeholder in place of
a real reading (most often yaw rate, occasionally speed or acceleration). Scaled into
engineering units these appear as physically impossible values (yaw rate ~327 deg/s,
speed ~163 m/s, acceleration ~20 m/s^2). This pass replaces out-of-physical-range sensor
values with NaN so the downstream feature pipeline imputes them instead of scoring them
as real measurements.

Per the paper's Section IV data-integrity note, these placeholders occur in under 0.1% of
rows in four of the five real-world datasets, and replacing them shifts every reported
AUROC by at most 0.002 -- below the display precision of the result tables. The default
benchmark reproduces the published tables without this pass; run it to reproduce the
Section IV robustness check. Row counts are unchanged (values are set to NaN, not dropped),
so the vehicle-disjoint split is identical before and after.

Usage:
    python -m benchmark.sanitize_sentinels --in raw.parquet --out clean.parquet
    python -m benchmark.sanitize_sentinels --in raw.csv     --out clean.csv
"""
import argparse

import numpy as np
import pandas as pd

# (lo, hi): sensor values strictly outside these physical ranges are the reserved
# "unavailable" sentinel leak and are replaced with NaN. Stopped traffic (speed 0,
# yaw 0) is legitimate and left untouched.
BOUNDS = {
    "speed_mps":       (0.0, 100.0),     # > 100 m/s = speed sentinel (~163)
    "heading_deg":     (-1.0, 360.5),    # > 360 = heading sentinel
    "accel_long_mps2": (-18.0, 18.0),    # |a| > 18 = acceleration sentinel (~20)
    "accel_lat_mps2":  (-18.0, 18.0),
    "yaw_rate_degs":   (-300.0, 300.0),  # |y| > 300 = yaw sentinel (~327)
    "yaw_rate_dps":    (-300.0, 300.0),  # accepted column alias
}


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with out-of-range sentinel values replaced by NaN."""
    df = df.copy()
    for field, (lo, hi) in BOUNDS.items():
        if field in df.columns:
            v = pd.to_numeric(df[field], errors="coerce")
            df[field] = v.where((v >= lo) & (v <= hi), np.nan)
    return df


def _read(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _write(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True, help="input .parquet or .csv")
    ap.add_argument("--out", dest="out", required=True, help="output path (same extension)")
    args = ap.parse_args()

    df = _read(args.inp)
    counts = {}
    for field, (lo, hi) in BOUNDS.items():
        if field in df.columns:
            v = pd.to_numeric(df[field], errors="coerce")
            n = int(((v < lo) | (v > hi)).sum())
            if n:
                counts[field] = n
    _write(sanitize_df(df), args.out)
    total = sum(counts.values())
    frac = 100.0 * total / max(len(df), 1)
    print(f"{args.inp} -> {args.out}  rows={len(df)}  "
          f"sentinels NaN'd: {counts or 'none'} (total {total}, {frac:.3f}% of rows)")


if __name__ == "__main__":
    main()
