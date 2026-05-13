"""Receiver grid construction for the KAD km 105 case study."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


def make_receiver_grid(
    road_line: LineString,
    half_width_m: float = 1000,
    spacing_m: float = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a square UTM receiver grid centered on the road midpoint.

    Returns ``grid_points`` as ``(N, 2)`` easting/northing coordinates,
    perpendicular distances to the road, and azimuths from the nearest point
    on the road toward each receiver. Azimuths use the propagation convention:
    degrees clockwise from north, direction toward the receiver.
    """
    if half_width_m <= 0:
        raise ValueError("half_width_m must be positive")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    midpoint = road_line.interpolate(0.5, normalized=True)
    x = np.arange(
        midpoint.x - half_width_m,
        midpoint.x + half_width_m + spacing_m * 0.5,
        spacing_m,
        dtype=float,
    )
    y = np.arange(
        midpoint.y - half_width_m,
        midpoint.y + half_width_m + spacing_m * 0.5,
        spacing_m,
        dtype=float,
    )
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    distances = np.empty(grid_points.shape[0], dtype=float)
    azimuths = np.empty(grid_points.shape[0], dtype=float)
    for i, (px, py) in enumerate(grid_points):
        receiver = Point(px, py)
        _, nearest_on_road = nearest_points(receiver, road_line)
        dx = px - nearest_on_road.x
        dy = py - nearest_on_road.y
        distances[i] = float(np.hypot(dx, dy))
        azimuths[i] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    return grid_points, distances, azimuths
