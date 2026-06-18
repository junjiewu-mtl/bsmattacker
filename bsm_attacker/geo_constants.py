"""
Unified geographic coordinate conversion constants.

All lat/lon ↔ metre conversions in the codebase should use these
constants instead of inline magic numbers.

Note: Haversine functions (which use Earth radius R = 6_371_000)
are NOT covered here — they are proper geodetic calculations and
should remain as-is.

References:
    - 1° latitude  ≈ 111,320 m (WGS-84 mean)
    - 1° longitude ≈ 111,320 * cos(lat) m
    - At 45.5°N (Montreal): 1° lon ≈ 77,963 m
"""

import math

import numpy as np

# Metres per degree of latitude (WGS-84 mean, constant everywhere)
DEG_LAT_TO_M: float = 111_320.0

# Earth mean radius in metres (for offset_coordinates-style calculations)
EARTH_RADIUS_M: float = 6_371_000.0


def deg_lon_to_m(lat_deg: float) -> float:
    """Metres per degree of longitude at a given latitude.

    Args:
        lat_deg: Latitude in degrees (scalar or array-like).

    Returns:
        Metres per degree of longitude. At the equator this equals
        DEG_LAT_TO_M; at the poles it approaches 0.

    Examples:
        >>> deg_lon_to_m(0.0)     # equator
        111320.0
        >>> deg_lon_to_m(45.5)    # Montreal
        77963.0...
    """
    return DEG_LAT_TO_M * math.cos(math.radians(lat_deg))


def deg_lon_to_m_vec(lat_deg_array) -> np.ndarray:
    """Vectorized version of deg_lon_to_m for numpy arrays.

    Args:
        lat_deg_array: Latitude(s) in degrees (numpy array).

    Returns:
        Metres per degree of longitude (numpy array, same shape).
    """
    return DEG_LAT_TO_M * np.cos(np.radians(lat_deg_array))


def positions_are_metres(lat_array) -> bool:
    """Detect whether the corpus uses Cartesian metres vs WGS-84 degrees.

    VeReMi-Ext synth stores SUMO local metres in lat/lon columns
    (max value typically thousands of metres). Real-site corpora store
    WGS-84 degrees (max abs value ≤ 90).

    Returns True iff the corpus is Cartesian metres — in which case
    position-attack injectors must NOT apply the deg→m conversion.
    """
    arr = np.abs(np.asarray(lat_array, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return float(arr.max()) > 90.0
