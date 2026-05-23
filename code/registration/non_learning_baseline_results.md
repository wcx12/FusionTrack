# Non-learning Baseline Results Summary

Primary fair-comparison protocol: `source-2 / crop-noise / 20-batch validation`.

Current protocol-aligned result: `code/registration/non_learning_source2_crop_protocol_results.md`

The threshold sweep remains robustness/appendix evidence, because it uses a broader train-split sweep rather than the exact learned-model validation protocol.

Remote run: `runs/mps_gaf_nonlearn_schema_source2_crop_eval20_full`

Dataset: `datasets/modelnet40_ply_hdf5_2048`

Split: `test`

Protocol: `source-2 / crop-noise / 20-batch validation`

Pairs: `40`

## Recommended Main Table

Threshold: `15 deg / 0.5`

| method | rot/deg | trans | chamfer | pose50 | success | skip | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `icp_point_to_point` | 24.868 | 0.244 | 0.024 | 37.083 | 0.475 | 0.000 | 0.0244 |
| `icp_point_to_plane` | 32.144 | 0.277 | 0.050 | 46.012 | 0.525 | 0.000 | 0.0207 |
| `icp_trimmed` | 27.303 | 0.274 | 0.046 | 41.015 | 0.300 | 0.000 | 0.0241 |
| `ransac_icp` | 45.844 | 0.369 | 0.037 | 64.269 | 0.225 | 0.000 | 0.6327 |
| `cpd` | 39.226 | 0.247 | 0.020 | 51.552 | 0.250 | 0.000 | 1.1773 |
| `identity` | 39.377 | 0.463 | 0.240 | 62.505 | 0.025 | 0.000 | 0.0000 |
| `fpfh_fgr` | 80.801 | 0.410 | 0.246 | 101.314 | 0.050 | 0.000 | 0.0453 |
| `fpfh_ransac` | 102.452 | 0.424 | 0.278 | 123.659 | 0.025 | 0.000 | 0.1980 |
| `gicp` | 62.958 | 0.345 | 0.313 | 80.187 | 0.000 | 0.000 | 0.0399 |
| `teaserpp` | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `super4pcs` | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `goicp` | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |

## Interpretation

- `icp_point_to_point` is the strongest practical baseline by overall pose/chamfer/runtime balance.
- `icp_point_to_plane` gives the highest relaxed success rate (`0.525`) but has worse mean rotation and Chamfer than point-to-point ICP.
- `cpd` gives the lowest Chamfer (`0.020`) but is much slower and has weaker pose accuracy than point-to-point ICP.
- `ransac_icp` reduces Chamfer compared with identity, but its rotation error and runtime make it a secondary baseline.
- `teaserpp`, `super4pcs`, and `goicp` are dependency skips, not accuracy failures.

## Robustness Appendix

The older train-split threshold sweep remains available as appendix evidence:

- Remote run: `runs/mps_gaf_nonlearn_threshold_sweep_v1`
- Cases: `base_crop`, `sparse_crop`, `dense_crop`, `points_512`, `jitter`, `clean`
- Pairs per threshold: `36`

## Authoritative Artifacts

- Protocol-aligned results: `code/registration/non_learning_source2_crop_protocol_results.md`
- MPS-GAF eval-schema summary: `code/registration/non_learning_mps_gaf_eval_schema_summary.json`
- Extended comparison-schema summary: `code/registration/non_learning_comparison_schema_summary.json`
- Main table markdown: `code/registration/non_learning_main_table.md`
- Main table CSV: `code/registration/non_learning_main_table.csv`
- Protocol run manifest: `code/registration/non_learning_protocol_run_manifest.json`
- External dependency probe: `code/registration/non_learning_external_dependency_probe.json`
- Protocol-aligned raw summary: `code/registration/non_learning_source2_crop_eval20_summary.json`
- Full threshold payload: `code/registration/threshold_sweep_payload.json`
- Full threshold tables: `code/registration/threshold_sweep_summary.md`
- Method research notes: `code/registration/non_learning_baseline_research.md`
