"""Road geometry helpers for the KAD km 105 case study."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString


def define_road_segment(
    center_lat: float,
    center_lon: float,
    azimuth_deg: float,
    length_m: float,
    utm_zone: str = "EPSG:32636",
) -> LineString:
    """Return a straight road segment as a UTM ``LineString``.

    ``azimuth_deg`` follows the usual GIS/navigation convention: degrees
    clockwise from north, pointing along the road axis. The KAD km 105 case
    uses 90 degrees as a compact east-west approximation.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive")

    transformer = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)

    theta = np.deg2rad(azimuth_deg)
    half_length = 0.5 * float(length_m)
    dx = np.sin(theta) * half_length
    dy = np.cos(theta) * half_length

    return LineString(
        [
            (center_x - dx, center_y - dy),
            (center_x + dx, center_y + dy),
        ]
    )
