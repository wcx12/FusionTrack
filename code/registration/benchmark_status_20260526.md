# Registration benchmark status - 2026-05-26

## Protocol

Current comparable ModelNet protocol:

- Dataset: `datasets/modelnet40_ply_hdf5_2048`
- Split: existing train/test split from the dataset package
- Evaluation transform: source-2, crop noise, `num_points=1024`, `num_sources_per_ref=2`
- Validation used during training: 20-batch validation protocol unless noted
- Main scalar: `pose_metric = RRE_mean + 50 * RTE_mean`

No project code/config in this pass should require hard-coded absolute paths.

## Metric definitions

All runners use transforms from source to reference.

- RRE: `acos(clamp((trace(R_gt^T R_pred) - 1) / 2, -1, 1))`, reported in degrees. Before computing RRE, rotation blocks are projected to SO(3) with SVD to avoid invalid values from slightly non-orthogonal predictions.
- RTE: `||t_pred - t_gt||_2`.
- Chamfer: after applying `T_pred` to source points, compute the bidirectional mean squared nearest-neighbor distance between transformed source and reference.
- RMSE columns use `sqrt(mean(error^2))`.
- Non-learning success rate uses the configured rotation/translation thresholds; failed method calls are counted as failed pairs instead of silently disappearing.

## ModelNet full-set results

Full-set learned evaluation has now run on 4936 pairs.

| method | pose | RRE | RTE | Chamfer | pairs |
|---|---:|---:|---:|---:|---:|
| MPS-GAF | 15.875 | 9.982 | 0.118 | 0.033 | 4936 |
| RPMNet | 16.242 | 10.428 | 0.116 | 0.036 | 4936 |
| DCP-DGCNN | 40.356 | 24.094 | 0.325 | 0.057 | 4936 |
| PointNetLK | 47.444 | 31.708 | 0.315 | 0.041 | 4936 |
| PRNet-DGCNN | 51.879 | 31.763 | 0.402 | 0.077 | 4936 |
| IDAM-GNN | 56.648 | 37.875 | 0.375 | 0.075 | 4936 |
| OMNet | 173.830 | 127.799 | 0.921 | 1.034 | 4936 |

The previous 20-batch learned evaluation was consistent with the full-set ordering:
MPS-GAF and RPMNet are clearly strongest; DCP/PointNetLK/PRNet/IDAM are mid-tier; OMNet is unstable under this schema.

## ModelNet 20-batch non-learning results

| method | pose | RRE | RTE | Chamfer | pairs |
|---|---:|---:|---:|---:|---:|
| ICP point-to-point | 34.662 | 22.619 | 0.241 | 0.024 | 160 |
| ICP point-to-plane | 36.604 | 25.401 | 0.224 | 0.059 | 160 |
| CPD rigid | 38.109 | 25.328 | 0.256 | 0.021 | 160 |
| Trimmed ICP | 38.148 | 24.658 | 0.270 | 0.041 | 160 |
| GICP | 50.216 | 30.384 | 0.397 | 0.174 | 160 |
| RANSAC-ICP | 61.778 | 43.671 | 0.362 | 0.035 | 160 |
| Identity | 64.910 | 40.984 | 0.479 | 0.257 | 160 |
| TurboReg | 71.235 | 53.875 | 0.347 | 0.055 | 160 |
| Super4PCS | 75.933 | 62.068 | 0.277 | 0.025 | 160 |
| FPFH-FGR | 86.295 | 64.276 | 0.440 | 0.052 | 160 |
| FPFH-RANSAC | 94.867 | 72.928 | 0.439 | 0.073 | 160 |
| TEASER++ | 100.463 | 77.713 | 0.455 | 0.076 | 160 |

Additional completion-pass smoke result on the new server:

| method | pose | RRE | RTE | Chamfer | pairs | note |
|---|---:|---:|---:|---:|---:|---|
| MPS-GAF short retrain eval20 | 37.619 | 23.293 | 0.287 | 0.027 | 80 | 3-epoch short retrain, 20 batches with `groups_per_batch=2`; use only as a path/schema check, not as the converged ModelNet result |

2026-05-27 expansion result for recent spatial-compatibility baselines:

| method | pose | RRE | RTE | Chamfer | pairs | note |
|---|---:|---:|---:|---:|---:|---|
| MAC-FPFH | 103.229 | 81.698 | 0.431 | 0.049 | 160 | project FPFH correspondence implementation of maximal-clique filtering; not official learned-descriptor MAC |
| SC2-PCR-FPFH | 90.881 | 71.574 | 0.386 | 0.038 | 160 | project FPFH correspondence implementation of second-order spatial compatibility; not official descriptor pipeline |
| KISS-Matcher | 114.931 | 86.175 | 0.575 | 0.170 | 160 | official `kiss-matcher` package wrapped in the project source-2/crop schema; not its official LiDAR benchmark setup |

On ModelNet source-2/crop, the best learned methods are not worse than the non-learning baselines:
MPS-GAF and RPMNet cut the pose metric by more than half relative to the best ICP/CPD results.
The weaker learned baselines are worse because these specific architectures are brittle under partial overlap/crop noise, not because the entire learned category is worse.
Super4PCS is now included as a standard robust classical baseline under the same source-2/crop eval20 schema. It is better than FPFH global-registration variants here, but still clearly weaker than ICP/CPD once the source and reference are already close enough for local refinement.
TurboReg and TEASER++ now also run under the same source-2/crop eval20 schema. They are useful robust-estimator references, but the current FPFH-correspondence input is weaker than the local ICP/CPD panel on this ModelNet protocol.
KISS-Matcher is now included as a recent robust global registration baseline. It installs cleanly from PyPI on the current server, but under this object-level source-2/crop protocol it is weaker than local ICP/CPD and the converged learned rows.
CPD has been rerun with pycpd similarity scale disabled, so the reported row is a rigid baseline rather than a similarity-transform baseline.

## Official-protocol external baselines

These runs are useful sanity checks, but they are not the same schema as the ModelNet source-2/crop runner above.

| dataset | method | pose | RRE | RTE | aggregation | extra metric | pairs | protocol note |
|---|---|---:|---:|---:|---|---|---:|---|
| ModelNet | GeoTransformer official | 2.570 | 1.620 | 0.019 | official pair summary | RR 0.490 | 2148 | official ModelNet protocol |
| 3DMatch | PointRegGPT GeoTransformer-16w | 12.921 | 5.621 | 0.146 | official pair summary | RR 0.967 | 1623 | PointRegGPT release checkpoint with GeoTransformer architecture |
| 3DMatch | PointRegGPT GeoTransformer-2w | 13.667 | 5.867 | 0.156 | official pair summary | RR 0.962 | 1623 | shorter PointRegGPT release checkpoint with GeoTransformer architecture |
| 3DMatch | GeoTransformer official rerun | 14.544 | 6.244 | 0.166 | official pair summary | RR 0.961 | 1623 | official 3DMatch protocol; rerun verified on 2026-05-26 |
| 3DMatch | RoITr official eval | n/a | 1.773 | 0.057 | mean of scene medians | weighted precision 0.941, IR 0.826, FMR 0.981 | 1623 | official RoITr 2500-correspondence RANSAC |
| 3DLoMatch | PointRegGPT GeoTransformer-16w | 44.056 | 19.256 | 0.496 | official pair summary | RR 0.816 | 1781 | PointRegGPT release checkpoint with GeoTransformer architecture |
| 3DLoMatch | PointRegGPT GeoTransformer-2w | 49.725 | 22.325 | 0.548 | official pair summary | RR 0.804 | 1781 | shorter PointRegGPT release checkpoint with GeoTransformer architecture |
| 3DLoMatch | GeoTransformer official rerun | 52.202 | 23.202 | 0.580 | official pair summary | RR 0.774 | 1781 | official 3DLoMatch protocol; rerun verified on 2026-05-26 |
| 3DLoMatch | RoITr official eval | n/a | 2.812 | 0.086 | mean of scene medians | weighted precision 0.764, IR 0.549, FMR 0.890 | 1781 | official RoITr 2500-correspondence RANSAC |

The 3DMatch and 3DLoMatch GeoTransformer official reruns are close to the converted-schema results below, which is a useful audit signal that the fixed-RRE conversion is not producing a materially different conclusion for GeoTransformer. RoITr's native official table uses a different RRE/RTE aggregation, so it is included as a native sanity check and converted into the project mean-RRE/RTE schema in the next table. The ModelNet official run should not be quoted as a direct win over MPS-GAF/RPMNet unless we also port it into the source-2/crop protocol, because the point sampling, split size, and perturbation protocol differ.

## 3DMatch and 3DLoMatch

Corrected fixed-RRE summaries:

| dataset | method | category | pose | RRE | RTE | Chamfer | pairs |
|---|---|---|---:|---:|---:|---:|---:|
| 3DMatch | PointRegGPT GeoTransformer-16w | learning | 12.701 | 5.387 | 0.146 | 0.383 | 1623 |
| 3DMatch | RoITr | learning | 12.850 | 5.038 | 0.156 | n/a | 1623 |
| 3DMatch | PointRegGPT GeoTransformer-2w | learning | 13.413 | 5.636 | 0.156 | 0.378 | 1623 |
| 3DMatch | GeoTransformer | learning | 14.310 | 6.010 | 0.166 | 0.382 | 1623 |
| 3DMatch | RPMNet | learning | 75.478 | 31.375 | 0.882 | 0.171 | 1623 |
| 3DMatch | DCP-DGCNN | learning | 76.861 | 31.472 | 0.908 | 0.234 | 1623 |
| 3DMatch | IDAM-GNN | learning | 82.627 | 34.003 | 0.972 | 0.231 | 1623 |
| 3DMatch | PointNetLK | learning | 83.937 | 34.018 | 0.998 | 0.203 | 1623 |
| 3DMatch | PRNet-DGCNN | learning | 84.948 | 34.222 | 1.015 | 0.200 | 1623 |
| 3DMatch | MPS-GAF no-normals | learning | 87.580 | 35.224 | 1.047 | 0.238 | 1623 |
| 3DMatch | Identity | non-learning | 87.909 | 35.154 | 1.055 | 0.468 | 1623 |
| 3DMatch | ICP trimmed | non-learning | 88.657 | 35.842 | 1.056 | 0.195 | 1623 |
| 3DMatch | GICP | non-learning | 89.642 | 33.711 | 1.119 | 0.587 | 1623 |
| 3DMatch | OMNet | learning | 90.445 | 37.295 | 1.063 | 0.551 | 1623 |
| 3DMatch | ICP point-to-point | non-learning | 90.941 | 37.234 | 1.074 | 0.157 | 1623 |
| 3DMatch | ICP point-to-plane | non-learning | 97.467 | 41.920 | 1.111 | n/a | 1623 |
| 3DMatch | MPS-GAF PPF | learning | 100.749 | 41.005 | 1.195 | 0.516 | 1623 |
| 3DMatch | CPD | non-learning | 104.048 | 47.067 | 1.140 | n/a | 1623 |
| 3DMatch | RANSAC-ICP | non-learning | 110.239 | 45.941 | 1.286 | n/a | 1623 |
| 3DMatch | KISS-Matcher | non-learning | 143.361 | 58.484 | 1.698 | 0.644 | 1623 |
| 3DMatch | FPFH-FGR | non-learning | 146.067 | 62.356 | 1.674 | n/a | 1623 |
| 3DMatch | FPFH-RANSAC | non-learning | 157.356 | 67.099 | 1.805 | n/a | 1623 |
| 3DMatch | SC2-PCR-FPFH | non-learning | 171.613 | 75.110 | 1.930 | 0.242 | 1623 |
| 3DMatch | MAC-FPFH | non-learning | 193.811 | 85.778 | 2.161 | 0.314 | 1623 |
| 3DMatch | TurboReg | non-learning | 204.595 | 87.252 | 2.347 | 0.605 | 1623 |
| 3DMatch | TEASER++ | non-learning | 236.522 | 99.907 | 2.732 | 0.648 | 1623 |
| 3DLoMatch | PointRegGPT GeoTransformer-16w | learning | 43.915 | 19.109 | 0.496 | 1.198 | 1781 |
| 3DLoMatch | RoITr | learning | 49.261 | 22.016 | 0.545 | n/a | 1781 |
| 3DLoMatch | PointRegGPT GeoTransformer-2w | learning | 49.570 | 22.182 | 0.548 | 1.199 | 1781 |
| 3DLoMatch | GeoTransformer | learning | 52.085 | 23.064 | 0.580 | 1.173 | 1781 |
| 3DLoMatch | DCP-DGCNN | learning | 132.011 | 58.091 | 1.478 | 0.344 | 1781 |
| 3DLoMatch | OMNet | learning | 138.322 | 61.125 | 1.544 | 0.770 | 1781 |
| 3DLoMatch | RPMNet | learning | 139.609 | 61.549 | 1.561 | 0.260 | 1781 |
| 3DLoMatch | PointNetLK | learning | 141.458 | 61.276 | 1.604 | 0.300 | 1781 |
| 3DLoMatch | PRNet-DGCNN | learning | 147.668 | 64.274 | 1.668 | 0.334 | 1781 |
| 3DLoMatch | Identity | non-learning | 148.168 | 61.350 | 1.736 | 0.687 | 1781 |
| 3DLoMatch | IDAM-GNN | learning | 151.918 | 65.645 | 1.725 | 0.399 | 1781 |
| 3DLoMatch | GICP | non-learning | 153.134 | 62.302 | 1.817 | 1.239 | 1781 |
| 3DLoMatch | ICP trimmed | non-learning | 158.922 | 68.140 | 1.816 | 0.378 | 1781 |
| 3DLoMatch | MPS-GAF no-normals | learning | 159.735 | 68.663 | 1.821 | 0.390 | 1781 |
| 3DLoMatch | ICP point-to-point | non-learning | 163.244 | 70.714 | 1.851 | 0.324 | 1781 |
| 3DLoMatch | CPD | non-learning | 166.458 | 74.235 | 1.844 | n/a | 1781 |
| 3DLoMatch | ICP point-to-plane | non-learning | 176.753 | 81.247 | 1.910 | n/a | 1781 |
| 3DLoMatch | RANSAC-ICP | non-learning | 181.840 | 80.131 | 2.034 | n/a | 1781 |
| 3DLoMatch | FPFH-FGR | non-learning | 182.996 | 80.094 | 2.058 | n/a | 1781 |
| 3DLoMatch | MPS-GAF PPF | learning | 185.626 | 82.079 | 2.071 | 0.826 | 1781 |
| 3DLoMatch | FPFH-RANSAC | non-learning | 199.276 | 89.255 | 2.200 | n/a | 1781 |
| 3DLoMatch | KISS-Matcher | non-learning | 245.428 | 109.374 | 2.721 | 1.182 | 1781 |
| 3DLoMatch | SC2-PCR-FPFH | non-learning | 255.500 | 118.516 | 2.740 | 0.370 | 1781 |
| 3DLoMatch | MAC-FPFH | non-learning | 259.660 | 119.976 | 2.794 | 0.475 | 1781 |
| 3DLoMatch | TurboReg | non-learning | 264.655 | 118.473 | 2.924 | 0.854 | 1781 |
| 3DLoMatch | TEASER++ | non-learning | 273.181 | 120.558 | 3.052 | 0.844 | 1781 |

Interpretation:

- The strong 3DMatch/3DLoMatch results are PointRegGPT GeoTransformer-16w, RoITr, and GeoTransformer, which are dataset-appropriate learned baselines.
- The PointRegGPT 2w checkpoint is consistently weaker than the 16w checkpoint on 3DMatch and 3DLoMatch, so shorter training/checkpoint maturity does matter. However, the already reported 16w checkpoint is converged enough to beat the non-learning panel by a wide margin on these datasets.
- The older learned baselines in the current table were trained under the ModelNet-style protocol and evaluated cross-domain on indoor fragments. Their weaker 3DMatch/3DLoMatch numbers should be treated as domain-mismatch evidence, not as a fair learned-vs-nonlearned conclusion.
- MPS-GAF with PPF features is worse than the no-normal variant on 3DMatch/3DLoMatch, which points to estimated-normal sensitivity under real fragment data.
- TurboReg is an ICCV 2025 learning-free robust estimator. In this project schema it is fed Open3D FPFH nearest-neighbor correspondences, not TurboReg's stronger official FCGF/PREDATOR correspondence bundles, so the rows above should be interpreted as a same-input robust-estimator check rather than TurboReg's official SOTA setting.

## ETH Laser Benchmark

ETH was added as a second independent extra benchmark. The run uses the standard four ETH scenes and all pairs from each scene's `gt.log`; no project-specific train/test split was invented. A direction audit on sample pairs confirmed that the log matrix maps the second fragment to the first fragment, which matches the runner's source-to-reference convention.

Full ETH result summary over 713 pairs:

| method | category | pose | RRE | RTE | Chamfer | pairs |
|---|---|---:|---:|---:|---:|---:|
| ICP point-to-point | non-learning | 130.479 | 56.305 | 1.483 | 2.491 | 713 |
| ICP point-to-plane | non-learning | 139.088 | 55.665 | 1.668 | 2.960 | 713 |
| RANSAC-ICP | non-learning | 140.915 | 58.550 | 1.647 | 2.669 | 713 |
| ICP trimmed | non-learning | 144.889 | 57.732 | 1.743 | 2.933 | 713 |
| Identity | non-learning | 147.959 | 59.179 | 1.776 | 3.113 | 713 |
| FPFH-RANSAC (5k cap) | non-learning | 147.959 | 59.179 | 1.776 | 3.113 | 713 |
| GICP | non-learning | 148.996 | 59.519 | 1.790 | 3.169 | 713 |
| MPS-GAF PPF | learning | 154.034 | 58.881 | 1.903 | 2.936 | 713 |
| IDAM-GNN | learning | 154.704 | 59.150 | 1.911 | 3.108 | 713 |
| RPMNet | learning | 154.968 | 57.363 | 1.952 | 2.716 | 713 |
| MPS-GAF no-normals | learning | 176.900 | 57.272 | 2.393 | 3.115 | 713 |
| FPFH-FGR | non-learning | 191.113 | 83.588 | 2.151 | 7.384 | 713 |
| PointNetLK | learning | 208.867 | 59.107 | 2.995 | 4.376 | 713 |
| TurboReg | non-learning | 237.778 | 107.603 | 2.603 | 11.737 | 713 |
| PointRegGPT GeoTransformer-2w | learning | 273.832 | 120.910 | 3.058 | 17.520 | 713 |
| GeoTransformer 3DMatch ckpt | learning | 287.792 | 119.915 | 3.358 | 17.397 | 713 |
| PointRegGPT GeoTransformer-16w | learning | 345.705 | 97.922 | 4.956 | 23.101 | 713 |
| TEASER++ | non-learning | 400.827 | 121.112 | 5.594 | 14.453 | 713 |
| DCP-DGCNN | learning | 578.540 | 66.512 | 10.241 | 66.377 | 713 |
| OMNet | learning | 596.354 | 60.455 | 10.718 | 68.805 | 713 |
| PRNet-DGCNN | learning | 820.083 | 91.273 | 14.576 | 103.957 | 713 |

Interpretation:

- These ETH learned results use 3DMatch-converged checkpoints as cross-benchmark evaluation, not ETH-specific training. ETH is a valid standard benchmark, but it does not provide the same train/val/test learning protocol as ModelNet or 3DMatch.
- GeoTransformer was additionally evaluated on all 713 official ETH `gt.log` pairs by converting those same pairs into the GeoTransformer data schema with 2048 sampled points per source/reference. The run initially hit CUDA OOM at pair 214 in one long process, then completed in chunks with the same output directory; the converted schema contains all 713 pairs.
- PointRegGPT GeoTransformer-16w and 2w were run through the same ETH conversion and chunked inference path. The 2w checkpoint is better than the 16w checkpoint on ETH pose because its RTE is lower, but both are still far behind ICP on this cross-dataset setting, so PointRegGPT's generated-data checkpoints should not be presented as ETH-generalization wins.
- A direct inverse-transform diagnostic on the ETH GeoTransformer feature dumps compared `pred vs gt`, `inv(pred) vs gt`, and `pred vs inv(gt)`. The inverse variants did not improve the pose metric for GeoTransformer or PointRegGPT, so the weak ETH learned rows are not explained by source/reference direction inversion.
- The high RRE values show ETH is still hard under the current uniform 2048-point downsampled schema. The useful comparison is the relative behavior under the same pair list and transform convention, not absolute SOTA performance.
- On ETH, simple point-to-point ICP is strongest among the currently completed group. This is different from ModelNet, where MPS-GAF/RPMNet are clearly better than ICP/CPD.
- RANSAC-ICP, FPFH-FGR, FPFH-RANSAC with a capped 5k budget, ICP point-to-plane, TurboReg, and TEASER++ now have full 713-pair ETH summaries. The capped FPFH-RANSAC result nearly matches identity, so it should not be presented as a fully tuned 100k-iteration RANSAC result. CPD remains too slow for this pass: even a capped 10-iteration full ETH attempt ran for more than 10 minutes without producing a summary and was stopped.

## Convergence check

Metric audit on 2026-05-26 found no source-to-reference direction bug or unit bug in the shared RRE/RTE/Chamfer formulas. The main formulas are:

- RRE: angle of `R_gt^T R_pred`, after SVD projection to SO(3).
- RTE: Euclidean distance between predicted and ground-truth translations.
- Chamfer: bidirectional mean squared nearest-neighbor distance after transforming source with `T_pred`.
- Pose metric: `RRE_mean + 50 * RTE_mean`.

Training summaries for the ModelNet converged checkpoints show that the best validation metric was followed by about 80 epochs without improvement for most models. The full-set evaluation also matches the 20-batch ordering. This makes a simple "not enough epochs" explanation unlikely for DCP, IDAM, PRNet, PointNetLK, RPMNet, and MPS-GAF.

For 3DMatch-specific training, the converged runs also stopped after 40 epochs without validation improvement for DCP, IDAM, OMNet, PointNetLK, PRNet, RPMNet, MPS-GAF without normals, and MPS-GAF with PPF. That makes "epoch not enough" unlikely for the currently reported 3DMatch learned-baseline numbers as well. The stronger explanation is domain/protocol mismatch: older ModelNet-oriented architectures degrade heavily on real indoor fragments, while GeoTransformer is designed and trained for this benchmark family.

PointRegGPT also provides a useful checkpoint-length sanity check: the 2w checkpoint is worse than 16w on both 3DMatch and 3DLoMatch, but both still remain far ahead of the learning-free FPFH/TurboReg/TEASER++ rows. On ETH, the 2w checkpoint has lower RTE and better pose than 16w, while the direction diagnostic still rules out a simple transform-inversion bug. This supports a narrower conclusion: under-training can explain part of the gap between two learned checkpoints on in-domain 3DMatch-family data, but the ETH behavior is dominated by cross-domain generalization rather than epoch count alone.

Evidence is recorded in the converged run summaries under `runs/*_converged_eval*/comparison_schema_summary.json` and the selected training summaries beside the corresponding checkpoints. The tables above use the fixed-RRE full-pair summaries where available.

Remaining caveats:

- Validation during training is still only the 20-batch protocol, so final claims should emphasize the full-set test results.
- OMNet remains suspiciously unstable and should be rechecked separately before using it as a central comparison.
- Dataset-specific training is needed before judging learned methods on 3DMatch/3DLoMatch against GeoTransformer.
- Older 40-pair summary JSON files are retained for provenance but should not be mixed with the 160-pair or full-pair tables as exact head-to-head results.

## Added dataset hooks and blockers

Newer-baseline status:

- Completed additional strong learning baseline: PointRegGPT GeoTransformer-16w and shorter 2w checkpoints from the PointRegGPT release package on 3DMatch and 3DLoMatch; both checkpoints also have completed ETH cross-dataset runs.
- Completed comparable new strong learning baselines: GeoTransformer (CVPR 2022) and RoITr (CVPR 2023), both on 3DMatch and 3DLoMatch.
- Completed a GeoTransformer cross-dataset ETH run on the official ETH pair list, using the 3DMatch checkpoint and the same 2048-point ETH schema used by the current project benchmark.
- Completed newer non-learning robust-estimator baseline: TurboReg (ICCV 2025), under the project FPFH-correspondence schema on ModelNet, 3DMatch, 3DLoMatch, and ETH.
- Completed additional classical robust baseline: TEASER++, under the project FPFH-correspondence schema on ModelNet, 3DMatch, 3DLoMatch, and ETH.
- Completed additional recent robust baseline: KISS-Matcher, under the project source-2/crop schema on ModelNet and under the official GeoTransformer/PREDATOR 3DMatch and 3DLoMatch metadata.
- Audited but not yet runnable in this environment: PARE-Net (ECCV 2024), CAST (NeurIPS 2024), BUFFER-X, RegTR, and JPCR. These are tracked as dependency/data/checkpoint blockers below, not as completed comparable results.

Standard extra-dataset coverage:

- 3DMatch benchmark family: uses the GeoTransformer/PREDATOR standard metadata and official 3DMatch / 3DLoMatch test pair lists. 3DLoMatch is reported separately because it is the standard low-overlap test split, not because a project-specific split was invented.
- ETH Laser Benchmark: uses the standard four ETH scenes and every pair from each scene's `gt.log`.
- KITTI odometry: metadata hooks use the GeoTransformer standard sequence split, but point clouds are absent, so KITTI is not yet counted as a completed extra dataset.

Implemented adapters:

- KITTI odometry pair metadata through the GeoTransformer-style metadata split.
- ETH laser registration pairs through standard scene logs.

Current blockers:

- KITTI point clouds are not present next to the available metadata.
- ETH point clouds are now available and evaluated for the current core method set.
- RegTR weights are available, but the current environment lacks MinkowskiEngine and PyTorch3D. The official stack targets an older Python/PyTorch/CUDA combination, and the available disk space is too tight for a clean isolated rebuild.
- PREDATOR / OverlapPredator official code is now staged, but its README targets Python 3.8, PyTorch 1.7.1, and CUDA 11.2. A compile probe on the current Python 3.12 server fails in `cpp_subsampling` and `cpp_neighbors` because `numpy.distutils` is absent. This is an old-stack extension blocker, not a GPU blocker.
- RoITr has a GitHub release asset for `model_3dmatch.pth`. The server-side `pointops` extension was patched for the removed THC include and now imports after `torch`; `tensorboardX` was installed. 3DMatch and 3DLoMatch inference plus official 2500-correspondence evaluations completed.
- PARE-Net remains blocked mainly by external pretrained-weight availability from Google Drive/Baidu-style mirrors and old-stack runtime requirements. The code is present, the official README lists 3DMatch/KITTI pretrained models, and a direct file-id `gdown` retry against the official Google Drive weight id failed with `Network is unreachable`; no checkpoint was produced.
- CAST is the strongest near-term new baseline candidate from the latest-method audit: the code is now present and the official release exposes `ckpt.zip`. A resumed checkpoint download reached a partial file but remained slow/flaky; runtime remains blocked because `pip install pytorch3d` has no wheel for the current Python 3.12/PyTorch 2.8 stack and MinkowskiEngine is absent. CAST also expects PREDATOR-style `.ply` fragments, while the current 3DMatch cache available to the benchmark is `.pth`.
- BUFFER-X is a new ICCV 2025 candidate already cloned. Its pure Python dependencies can install on the current server, but full inference is blocked because `nvcc` is absent and the required `pointnet2_ops` CUDA extension cannot be built. `torch_batch_svd` is also still absent after the failed extension phase. A Hugging Face model-repo listing attempt for `Hyungtae-Lim/BUFFER-X` hung until interrupted, and the default Dropbox source remains unreliable from this server.
- BUFFER-X full cross-dataset suite remains blocked by dataset size; the README states the generalization benchmark data requires about 130 GB, which does not fit the current data disk.
- INTEGER (NeurIPS 2024) was added to the candidate audit as a recent unsupervised outdoor method, but it targets KITTI/nuScenes and still needs separate environment/checkpoint staging, so it remains an empty record for this pass.
- JPCR code is present, but it targets a separate Jittor/vision3d runtime and Google Drive-hosted pretrained weights. Jittor and vision3d are absent from the current environment.
- PointRegGPT code is present and its GitHub release weights package has been staged. The GeoTransformer-16w and GeoTransformer-2w checkpoints now run through a lightweight export path that avoids multi-GB feature dumps; full 3DMatch and 3DLoMatch converted-schema summaries are complete. The CoFiNet path has data and weights staged, but its official stack is blocked in the current server environment: requirements pin Torch 1.8/CUDA 11.1, `cpp_subsampling` and `cpp_neighbors` import the removed `numpy.distutils`, and `grouping_cuda` includes the removed PyTorch `THC/THC.h` header. This is a Python/PyTorch extension compatibility issue, not a GPU-availability issue.
