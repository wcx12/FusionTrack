# Registration Benchmark Protocol Sources - 2026-05-27

This note records the external method/protocol sources used to scope the
benchmark expansion. Filesystem paths in project records remain
repository-relative.

## Baseline Methods

| method | source | benchmark use |
|---|---|---|
| GeoTransformer | CVPR 2022, official repository: `https://github.com/qinzheng93/GeoTransformer` | Keep as a learned top-venue baseline on its official 3DMatch/3DLoMatch/KITTI/ModelNet protocols and converted project schema where already staged. |
| PREDATOR / OverlapPredator | CVPR 2021 Oral, official repository: `https://github.com/prs-eth/OverlapPredator` | Use its standard processed data/metadata line for 3DMatch/3DLoMatch; do not invent a new split. |
| MAC / maximal cliques | CVPR 2023 / TPAMI extension, official repository: `https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques` | Project implementation is `MAC-FPFH`, using FPFH nearest-neighbor correspondences plus greedy maximal-clique filtering. Report as same-input FPFH variant, not official learned-descriptor MAC. |
| SC2-PCR | CVPR 2022, official repository: `https://github.com/ZhiChen902/SC2-PCR` | Project implementation is `SC2-PCR-FPFH`, using FPFH correspondences plus second-order spatial compatibility scoring. Report as same-input FPFH variant. |
| KISS-Matcher | ICRA 2025 / Quatro++ IJRR 2024 line, official repository: `https://github.com/MIT-SPARK/KISS-Matcher` | Added as an optional `kiss-matcher` package wrapper. Completed ModelNet source-2/crop, 3DMatch, and 3DLoMatch rows are project-schema checks, not the official LiDAR benchmark setup. |
| CAST | NeurIPS 2024, official repository: `https://github.com/RenlangHuang/CAST` | Candidate recent learned baseline; dependency-gated because the current server lacks `nvcc` and old extension stack support. |
| BUFFER-X | ICCV 2025 Highlight, official repository: `https://github.com/MIT-SPARK/BUFFER-X` | Candidate zero-shot learned baseline; full official evaluation is skipped because it needs CUDA extensions and large preprocessed data beyond the current disk budget. |
| INTEGER | NeurIPS 2024, official repository: `https://github.com/kezheng1204/INTEGER` | Candidate unsupervised outdoor baseline; skipped because it targets KITTI/nuScenes and needs separate environment/checkpoint staging. |
| RoITr | CVPR 2023, official repository: `https://github.com/haoyu94/RoITr` | Keep existing 3DMatch/3DLoMatch converted-schema rows; ETH remains dependency/runtime gated unless the staged RoITr environment is reusable. |

## Dataset Protocols

| dataset | protocol decision |
|---|---|
| ModelNet40 | Use the existing package train/test split and the project source-2/crop/20-batch protocol for same-schema comparison. Do not compare official ModelNet protocol rows as direct wins/losses against source-2/crop rows. |
| 3DMatch | Use PREDATOR/GeoTransformer official metadata and processed fragments. No project-specific train/test split. |
| 3DLoMatch | Use the official low-overlap benchmark metadata from the 3DMatch family. No project-specific train/test split. |
| ETH | Use the four-scene ETH laser benchmark `gt.log` pairs for generalization tests; do not create a train/val/test split. |
| KITTI | If point clouds are staged, use the common GeoTransformer-style odometry protocol with train sequences 00-05, validation 06-07, and test 08-10. |

## Reporting Rules

- Learned methods are only claimed as converged after early-stop/plateau evidence
  or a completed epoch budget plus final evaluation.
- Methods requiring missing CUDA extensions, old Python/Torch stacks, or unavailable
  checkpoints are dependency-gated, not counted as failed accuracy.
- Project FPFH variants of MAC and SC2-PCR are valid same-input baseline rows, but
  must not be described as official learned-descriptor results.
- If the PREDATOR point-cloud package lacks pair metadata, fetch GeoTransformer's
  official `3DMatch.pkl` and `3DLoMatch.pkl`; do not synthesize a custom split.
