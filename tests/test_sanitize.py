"""Data-integrity pass: sentinel values become NaN, legitimate readings are untouched."""
import numpy as np
import pandas as pd

from benchmark.sanitize_sentinels import sanitize_df


def test_sentinels_become_nan_and_rows_are_preserved():
    df = pd.DataFrame(
        {
            "device_id":       ["V1", "V1", "V1", "V1"],
            "speed_mps":       [14.9, 163.0, 20.0, 0.0],    # row 1: speed sentinel
            "heading_deg":     [90.0, 90.0, 361.0, 45.0],   # row 2: heading sentinel
            "accel_long_mps2": [0.20, 0.10, 0.00, 20.5],    # row 3: accel sentinel
            "yaw_rate_degs":   [0.10, 327.0, 0.00, -0.20],  # row 1: yaw sentinel
        }
    )
    out = sanitize_df(df)

    # out-of-range sentinels are replaced with NaN
    assert np.isnan(out.loc[1, "speed_mps"])
    assert np.isnan(out.loc[2, "heading_deg"])
    assert np.isnan(out.loc[3, "accel_long_mps2"])
    assert np.isnan(out.loc[1, "yaw_rate_degs"])

    # legitimate readings (including stopped traffic: speed 0, yaw 0) are untouched
    assert out.loc[0, "speed_mps"] == 14.9
    assert out.loc[0, "yaw_rate_degs"] == 0.10
    assert out.loc[3, "speed_mps"] == 0.0
    assert out.loc[2, "yaw_rate_degs"] == 0.0

    # no rows are dropped and the input frame is not mutated in place
    assert len(out) == len(df)
    assert df.loc[1, "speed_mps"] == 163.0


def test_yaw_rate_dps_alias_is_handled():
    df = pd.DataFrame({"yaw_rate_dps": [0.1, 327.0, -5.0]})
    out = sanitize_df(df)
    assert out["yaw_rate_dps"].isna().tolist() == [False, True, False]
