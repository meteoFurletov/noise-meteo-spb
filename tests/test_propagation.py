import numpy as np

from src.case_study.propagation import (
    compute_geometric_divergence,
    derive_c0_from_pfav,
)


def test_derive_c0_from_pfav_uses_distance_dependent_delta_l():
    c0 = derive_c0_from_pfav(0.5, distance_m=500.0)

    # delta_L(500 m) = 3 + 6 log10(500 / 50) = 9 dB.
    expected = -10.0 * np.log10(0.5 + 0.5 * 10.0 ** (-9.0 / 10.0))
    assert np.isclose(c0, expected)
    assert np.isclose(c0, 2.495330536)


def test_geometric_divergence_transitions_from_line_to_point_source():
    distances = np.asarray([50.0, 200.0, 500.0])
    a_div = compute_geometric_divergence(distances, road_length_m=2000.0)

    assert np.all(np.diff(a_div) > 0.0)
    assert np.allclose(
        a_div,
        10.0 * np.log10(2.0 * np.pi * distances * (1.0 + 4.0 * distances / 2000.0)),
    )
