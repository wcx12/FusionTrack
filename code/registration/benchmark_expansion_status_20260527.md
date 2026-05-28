# Benchmark Expansion Status - 2026-05-27

Scope: expand the registration benchmark with newer/common top-venue baselines and
at least two standard extra datasets, while keeping all filesystem paths
repository-relative in records and commands.

## Active Server Work

| job | status | evidence | note |
|---|---|---|---|
| MPS-GAF ModelNet40 source-2/crop convergence run | running | `runs/mps_gaf_modelnet_source2_crop_converged_20260527` | GPU is active. No `max_train_steps`; validation uses 20 batches, early stopping, and LR plateau. |
| MPS-GAF final eval watcher | waiting | `runs/mps_gaf_modelnet_source2_crop_converged_eval_20260527` | Waits for the convergence run to exit, then evaluates the best checkpoint with the same source-2/crop/20-batch protocol. |
| PREDATOR official data package download | done | `runs/data_downloads/predator_data_aria2.log` | This is the standard 3DMatch/3DLoMatch data package used by the PREDATOR line; no custom split is introduced. |
| Auto-launch 3DMatch/3DLoMatch MAC/SC2/KISS | done | `runs/predator_standard_20260527/auto_launch.log` | Uses official GeoTransformer metadata with PREDATOR point clouds and launched `mac,sc2_pcr,kiss_matcher`. |
| MAC/SC2-PCR ModelNet40 eval20 | done | `runs/mac_sc2_modelnet_source2_crop_eval20_20260527` | CPU run, same source-2/crop/20-batch schema. |
| MAC/SC2-PCR ModelNet40 eval20 g4 | done | `runs/mac_sc2_modelnet_source2_crop_eval20_g4_20260527` | CPU run matching the 160-pair non-learning main-table protocol. |
| KISS-Matcher ModelNet40 eval20 g4 | done | `runs/kiss_modelnet_source2_crop_eval20_g4_20260527` | CPU run matching the 160-pair non-learning main-table protocol. |

## Added Baselines

| method key | source line | implementation status | reporting caveat |
|---|---|---|---|
| `mac`, `mac_fpfh` | MAC / maximal-clique robust registration, CVPR 2023 and TPAMI extension | implemented in `code/registration/non_learning_baselines.py` | Uses project FPFH nearest-neighbor correspondences plus greedy maximal-clique filtering, so report as `MAC-FPFH`, not official learned-descriptor MAC. |
| `sc2_pcr` | SC2-PCR spatial compatibility, CVPR 2022 | implemented in `code/registration/non_learning_baselines.py` | Uses project FPFH correspondences plus second-order spatial compatibility scoring, so report as `SC2-PCR-FPFH`. |
| `kiss_matcher` | KISS-Matcher, ICRA 2025 / Quatro++ IJRR 2024 line | implemented as an optional `kiss-matcher` package wrapper | Uses the project point-cloud inputs and source-2/crop protocol, not KISS-Matcher's official LiDAR benchmark setup. |

## Server Smoke Result

Command class: `run_registration_benchmark.py` with `modelnet40`, source-2,
crop-noise, 1 evaluation batch, 256 points, `mac,sc2_pcr`.

Artifact synced locally:
`code/registration/server_artifacts/20260527_expansion/runs/mac_sc2_modelnet_source2_crop_smoke_20260527/comparison_schema_summary.json`

| method | pose50 | RRE | RTE | Chamfer | pairs | note |
|---|---:|---:|---:|---:|---:|---|
| `mac` | 47.143 | 38.197 | 0.179 | 0.047 | 2 | schema smoke only |
| `sc2_pcr` | 103.879 | 88.332 | 0.311 | 0.046 | 2 | schema smoke only |
| `kiss_matcher` | 107.506 | 88.039 | 0.389 | 0.175 | 2 | schema smoke only |

## New ModelNet40 Eval20 Result

Command class: `run_registration_benchmark.py` with `modelnet40`, source-2,
crop-noise, 20 evaluation batches, 1024 points, `mac,sc2_pcr`.

Artifact synced locally:
`code/registration/server_artifacts/20260527_expansion/runs/mac_sc2_modelnet_source2_crop_eval20_20260527/comparison_schema_summary.json`

| method | pose50 | RRE | RTE | Chamfer | pairs | success | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mac` | 104.116 | 85.563 | 0.371 | 0.051 | 40 | 0.350 | 0.0994 |
| `sc2_pcr` | 78.310 | 63.257 | 0.301 | 0.039 | 40 | 0.425 | 0.0412 |

The 40-pair run above was kept as an initial completion artifact. The main
non-learning table uses the 160-pair variant below because it matches the
existing corrected eval20 protocol.

Artifact synced locally:
`code/registration/server_artifacts/20260527_expansion/runs/mac_sc2_modelnet_source2_crop_eval20_g4_20260527/comparison_schema_summary.json`

| method | pose50 | RRE | RTE | Chamfer | pairs | success | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mac` | 103.229 | 81.698 | 0.431 | 0.049 | 160 | 0.350 | 0.0704 |
| `sc2_pcr` | 90.881 | 71.574 | 0.386 | 0.038 | 160 | 0.362 | 0.0426 |
| `kiss_matcher` | 114.931 | 86.175 | 0.575 | 0.170 | 160 | 0.075 | 0.0962 |

## New 3DMatch Standard Result

Command class: `run_registration_benchmark.py` with official GeoTransformer
`3DMatch.pkl`, PREDATOR indoor point clouds, 2048 sampled points, and
`mac,sc2_pcr,kiss_matcher`.

Artifact synced locally:
`code/registration/server_artifacts/20260527_expansion/runs/predator_standard_20260527/3dmatch_mac_sc2_pcr_kiss_matcher/comparison_schema_summary.json`

| method | pose50 | RRE | RTE | Chamfer | pairs | success | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mac` | 193.811 | 85.778 | 2.161 | 0.314 | 1623 | 0.261 | 0.0939 |
| `sc2_pcr` | 171.613 | 75.110 | 1.930 | 0.242 | 1623 | 0.280 | 0.1018 |
| `kiss_matcher` | 143.361 | 58.484 | 1.698 | 0.644 | 1623 | 0.367 | 0.1076 |

## New 3DLoMatch Standard Result

Command class: `run_registration_benchmark.py` with official GeoTransformer
`3DLoMatch.pkl`, PREDATOR indoor point clouds, 2048 sampled points, and
`mac,sc2_pcr,kiss_matcher`.

Artifact synced locally:
`code/registration/server_artifacts/20260527_expansion/runs/predator_standard_20260527/3dlomatch_mac_sc2_pcr_kiss_matcher/comparison_schema_summary.json`

| method | pose50 | RRE | RTE | Chamfer | pairs | success | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mac` | 259.660 | 119.976 | 2.794 | 0.475 | 1781 | 0.025 | 0.0880 |
| `sc2_pcr` | 255.500 | 118.516 | 2.740 | 0.370 | 1781 | 0.022 | 0.1034 |
| `kiss_matcher` | 245.428 | 109.374 | 2.721 | 1.182 | 1781 | 0.046 | 0.1044 |

## Standard Dataset Protocols

| dataset | protocol decision |
|---|---|
| 3DMatch | Use official/PREDATOR/GeoTransformer metadata. 3DMatch test pairs follow the common overlap-positive benchmark, not a custom split. |
| 3DLoMatch | Use official low-overlap metadata from the same benchmark family, not a custom split. |
| ETH | Use the four-scene ETH laser benchmark with `gt.log` pairs as a generalization test. |
| KITTI | Use GeoTransformer-style odometry metadata if point clouds can be staged within disk limits; common split is sequences 00-05 train, 06-07 validation, 08-10 test. |

## Current Constraints

- Current server has no `nvcc`; methods requiring CUDA extension builds remain dependency-gated.
- Full learned convergence is only claimable after early stop/plateau evidence or completed epoch budget plus final eval.
- PREDATOR data download was switched from single-connection `wget` to resumable multi-connection `aria2c`; if interrupted, resume the same archive instead of restarting.
- The downloaded PREDATOR data package did not include `3DMatch.pkl` or `3DLoMatch.pkl`, so the auto-launcher now downloads the official GeoTransformer metadata instead of generating a custom split.
- Overly complex methods or datasets can remain as empty records and should not block the current benchmark.

## Latest Monitoring

| item | status | evidence |
|---|---|---|
| MPS-GAF ModelNet40 source-2/crop convergence run | running, not converged | latest epoch 93; epochs_since_best 6; best pose metric 1.654, best RRE 1.090, best RTE 0.0113, best Chamfer 0.0422 over 80 validation pairs |
| MPS-GAF final eval watcher | waiting | `auto_eval_status.jsonl` reports the train process is still alive; no final eval summary yet |
| 3DMatch MAC/SC2/KISS | done | full 1623-pair CPU evaluation completed and was synced into `server_artifacts` |
| 3DLoMatch MAC/SC2/KISS | done | full 1781-pair CPU evaluation completed and was synced into `server_artifacts` |

## Skipped / Empty Records

These entries are intentionally left empty under the simplified scope. They are
not counted as failed accuracy results.

| item | status | reason |
|---|---|---|
| `CAST` | empty | Requires a heavier CUDA-extension/dependency stack than the current server provides. |
| `CoFiNet` | empty | Requires separate official environment/checkpoint staging; skip unless already available. |
| `RegTR` | empty | Requires separate official environment/checkpoint staging; skip unless already available. |
| `PointDSC` | empty | CUDA-extension dependency risk is high on a server without `nvcc`. |
| `BUFFER-X` | empty | ICCV 2025 zero-shot method, but full evaluation requires CUDA extensions plus large official preprocessed data beyond the current disk budget. |
| `INTEGER` | empty | NeurIPS 2024 unsupervised outdoor method; separate cleaned environment/checkpoint staging is still required. |
| official MAC descriptor pipeline | empty | Current completed result is the same-schema `MAC-FPFH` variant. |
| official SC2-PCR pipeline | empty | Current completed result is the same-schema `SC2-PCR-FPFH` variant. |
| KITTI | empty | Data is not staged and disk budget is limited; skip until a dedicated data volume is available. |
| ETH full-budget CPD | empty | The capped ETH CPD attempt was already too slow, so the full-budget run is skipped. |
