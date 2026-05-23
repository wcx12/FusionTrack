# Non-learning Main Table

Protocol: `source-2 / crop-noise / 20-batch validation`

Remote run: `runs/mps_gaf_nonlearn_schema_source2_crop_eval20_full`

| method | pairs | rot/deg | trans | chamfer | pose50 | success | skip | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `icp_point_to_point` | 40 | 24.868 | 0.244 | 0.024 | 37.083 | 0.475 | 0.000 | 0.0244 |
| `icp_point_to_plane` | 40 | 32.144 | 0.277 | 0.050 | 46.012 | 0.525 | 0.000 | 0.0207 |
| `icp_trimmed` | 40 | 27.303 | 0.274 | 0.046 | 41.015 | 0.300 | 0.000 | 0.0241 |
| `ransac_icp` | 40 | 45.844 | 0.369 | 0.037 | 64.269 | 0.225 | 0.000 | 0.6327 |
| `cpd` | 40 | 39.226 | 0.247 | 0.020 | 51.552 | 0.250 | 0.000 | 1.1773 |
| `identity` | 40 | 39.377 | 0.463 | 0.240 | 62.505 | 0.025 | 0.000 | 0.0000 |
| `fpfh_fgr` | 40 | 80.801 | 0.410 | 0.246 | 101.314 | 0.050 | 0.000 | 0.0453 |
| `fpfh_ransac` | 40 | 102.452 | 0.424 | 0.278 | 123.659 | 0.025 | 0.000 | 0.1980 |
| `gicp` | 40 | 62.958 | 0.345 | 0.313 | 80.187 | 0.000 | 0.000 | 0.0399 |
| `teaserpp` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `super4pcs` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `goicp` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |

Notes:

- `pose50` is `rotation_error_deg_mean + 50 * translation_error_mean`.
- `success` uses `15 deg / 0.5`.
- `teaserpp`, `super4pcs`, and `goicp` are dependency/wrapper skips.
