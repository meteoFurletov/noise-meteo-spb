# Fig 3 Propagation Diagnostic

Run on 2026-05-13 after applying the distance-dependent `p_fav -> C0` model and finite-line `A_div` update.

## Setup

- Source: KAD 105 km, 59.999980 N, 30.476457 E
- UTM 36N source coordinate: x=359265.284 m, y=6654093.661 m
- Nearest climatology cell: 18 (`cell_lat=60.000000`, `cell_lon=30.500000`)
- Case: DJF-night
- Emission level from current MGSU helper: `85.636 dBA`
- Maximum-favorability sector: sector 1, propagation azimuth 20.0 deg, `p_fav=0.768`, `C0_500m=0.985 dB`
- Minimum-favorability sector: sector 10, propagation azimuth 200.0 deg, `p_fav=0.506`, `C0_500m=2.455 dB`

## A_div Check
| Distance, m | A_div, dB | delta_L, dB |
|---:|---:|---:|
| 50 | 25.39 | 3.00 |
| 200 | 32.45 | 6.61 |
| 500 | 37.98 | 9.00 |

## 55 dBA Isoline Radius, DJF-Night
| Direction | Azimuth, deg | Radius, m |
|---|---:|---:|
| N | 0.0 | 102.1 |
| NE | 45.0 | 102.3 |
| E | 90.0 | 99.8 |
| SE | 135.0 | 94.0 |
| S | 180.0 | 91.7 |
| SW | 225.0 | 91.3 |
| W | 270.0 | 92.7 |
| NW | 315.0 | 97.6 |

Max/min radius ratio: `1.120`.

## Levels Along Maximum-Favorability Direction
Direction: propagation azimuth 20.0 deg.

| Distance, m | Level, dBA |
|---:|---:|
| 50 | 59.34 |
| 200 | 50.44 |
| 500 | 43.03 |

## Levels Along Minimum-Favorability Direction
Direction: propagation azimuth 200.0 deg.

| Distance, m | Level, dBA |
|---:|---:|
| 50 | 58.93 |
| 200 | 49.32 |
| 500 | 41.62 |

At 500 m, maximum minus minimum favorability is `1.41 dBA`.

## Pass/Fail Against Requested Gates
- Ballpark levels: `False`
- 55 dBA max/min radius ratio >= 1.4: `False`
- 500 m max-min contrast 3-5 dBA: `False`

## Diagnosis

The requested correction increases the distance-dependent `delta_L`, but it still does not produce the target DJF-night directional contrast.
The plumbing is still per-sector: `p_fav` ranges from 0.506 to 0.768 for DJF-night, and those sector values are used directly in `derive_c0_from_pfav(p_sector[idx], distance)`.
The limiting factor is the two-state energy mixture itself. Even as `delta_L -> infinity`, `C0 = -10 log10(p_fav)`, so this `p_fav` range can only create about 1.8 dB of sector-to-sector `C0` spread before the GOST `C_met` distance factor.
At 500 m the GOST geometry factor is `1 - 20/500 = 0.96`, so it is not the main suppressor; the `p_fav -> C0` mapping is.
Mode C is not averaging sectors; the sector lookup is directional and per receiver. To reach 3-5 dBA contrast, the model needs either a different mapping than `C0 = L_fav - L_long_term` from `p_fav`, a larger directional `p_fav` spread, or an additional directional propagation term beyond `C_met`.

Per the gate in the request, Fig 3 was not regenerated after this failed diagnostic.
