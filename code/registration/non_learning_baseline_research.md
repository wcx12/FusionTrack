# Non-learning Registration Baseline Research Notes

Date: 2026-05-23

## Scope

This note is limited to non-learning or mostly classical point-cloud registration baselines. The current benchmark intentionally excludes trained MPS-GAF weights and deep registration models.

## Practical baseline set

The first benchmark phase should keep these methods fixed:

| group | methods | status in benchmark | reason |
|---|---|---|---|
| Local refinement | point-to-point ICP, point-to-plane ICP | implemented | Standard local alignment baselines. Strong only when initialization is near the solution. |
| Robust ICP variants | trimmed ICP, RANSAC + ICP | implemented | Directly tests robustness to partial overlap, crop, jitter, and outliers. |
| Feature + global initialization | FPFH + RANSAC, FPFH + FGR | implemented | Widely used classical global-registration pipeline and available through Open3D. |
| Probabilistic / dense | CPD, GICP | implemented | Useful non-learning alternatives with different objective assumptions. |
| No-registration floor | identity | implemented | Required sanity baseline to show whether a method actually improves alignment. |
| Spatial compatibility filtering | MAC-FPFH, SC2-PCR-FPFH | implemented | Recent correspondence-graph robust estimators that can run without CUDA extensions when paired with FPFH correspondences. |
| External robust/global | TEASER++, TurboReg, KISS-Matcher, Super4PCS, Go-ICP | TEASER++, TurboReg, and KISS-Matcher implemented; Super4PCS and Go-ICP optional | Strong classical references, but some variants need external C++/binding setup. |
| Voxel distribution | NDT | phase-2 candidate | Widely used in robotics/SLAM, but not included in phase 1 because the current Python/Open3D stack has no stable NDT registration implementation. |

## Literature-backed method map

- ICP remains the canonical local baseline. Use point-to-point and point-to-plane variants as separate rows because they fail differently under normals, partial overlap, and poor initialization. Open3D documents both point-to-point and point-to-plane ICP registration pipelines: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
- FPFH + RANSAC is the standard feature-based coarse registration baseline. FPFH was introduced for 3D registration in ICRA 2009, and Open3D uses FPFH features in its global registration tutorial: https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf and https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
- Fast Global Registration (FGR) is a strong classical global method for partially overlapping surfaces, published at ECCV 2016. It optimizes a robust objective over candidate matches without inner-loop closest-point updates: https://vladlen.info/publications/fast-global-registration/
- TEASER++ is a key robust, certifiable registration reference from RSS/T-RO era work. It is especially relevant when correspondences contain many outliers, but its Python/C++ dependency path should stay optional in this repository: https://github.com/MIT-SPARK/TEASER-plusplus
- TurboReg is a recent ICCV 2025 learning-free robust estimator for PCR. It supports CPU/GPU execution and reports strong official 3DMatch/KITTI recall when paired with stronger correspondence sources. In this project it is run through the same FPFH-correspondence path as the other classical global methods, so the result is a same-input robust-estimator check rather than TurboReg's official descriptor setting: https://github.com/Laka-3DV/TurboReg
- MAC / maximal-clique registration is a recent robust correspondence selection direction from CVPR 2023 and a later TPAMI extension. The project implementation is `mac_fpfh`: FPFH nearest-neighbor correspondences plus a greedy maximal-clique compatibility graph, so report it as a same-input FPFH variant rather than the official learned-descriptor setting: https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques
- SC2-PCR is a CVPR 2022 spatial compatibility baseline. The project implementation is `sc2_pcr_fpfh`: FPFH correspondences plus second-order spatial compatibility scoring, again intended as a reproducible same-input robust estimator rather than an official descriptor pipeline: https://github.com/ZhiChen902/SC2-PCR
- Super4PCS is a classic global alignment method for arbitrary initial poses and low-overlap cases. It is a useful external dependency candidate after the Open3D/PyCPD baselines are stable: https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf
- Go-ICP is a globally optimal branch-and-bound ICP formulation. It is important as a classical reference, but usually slower and dependency-heavy, so it is better treated as optional external validation: https://arxiv.org/abs/1605.03344
- CPD models registration probabilistically and supports rigid and non-rigid settings. The current phase only uses rigid CPD: https://arxiv.org/abs/0905.2635
- NDT is a widely used classical registration family in robotics and mapping, but it usually comes through PCL, Autoware, or SLAM-specific stacks rather than the current Python-only benchmark stack. Keep it as a phase-2 dependency candidate instead of adding a partial reimplementation.
- Recent surveys show the field's top-venue frontier has moved heavily toward deep registration, but traditional methods are still necessary for reproducible non-learning baselines and ablation floors. A 2024 Pattern Recognition survey covers rigid pairwise registration broadly, while IJCAI 2024 focuses on deep-learning point-cloud registration taxonomy: https://www.sciencedirect.com/science/article/pii/S0031320324001596 and https://www.ijcai.org/proceedings/2024/922
- KISS-Matcher is a recent open-source classical/geometry-heavy direction that revisits FPFH-style matching with graph pruning. It now runs through the official `kiss-matcher` Python package in the project benchmark, so report it as a measured project-schema baseline rather than as the official LiDAR benchmark setup: https://arxiv.org/abs/2409.15615

## Implementation recommendation

For thesis-quality reporting, keep the first table to methods that can be executed now under one reproducible Python environment:

1. identity
2. point-to-point ICP
3. point-to-plane ICP
4. trimmed ICP
5. RANSAC + ICP
6. FPFH + RANSAC
7. FPFH + FGR
8. GICP
9. rigid CPD
10. MAC-FPFH
11. SC2-PCR-FPFH
12. KISS-Matcher

Keep NDT out of the first table unless a stable dependency is installed and it can emit the same metrics without changing the benchmark protocol.

Report Super4PCS and Go-ICP as "configured but dependency unavailable" unless their external builds are installed and the same script can execute them. Do not compare missing external methods as failed accuracy results; report them as skipped dependency cases. TEASER++, TurboReg, and KISS-Matcher now have runnable wrappers and should be reported as measured baselines when their project-relative dependencies are present.

## Current dependency status

The remote environment currently has the Python dependencies needed for the running phase-1 baselines:

- `open3d`: available
- `pycpd`: available
- `teaserpp_python`: available from the project-relative TEASER++ build path
- `turboreg_gpu`: available after installing the project-relative TurboReg binding
- `kiss-matcher`: available as a Python package on the current run server

The following external methods are not available yet:

- `super4pcs`
- `goicp`

These are not simple missing imports in the same sense as a pure Python package. A remote probe found no matching pip distributions for `super4pcs` or `pygoicp`. Super4PCS and Go-ICP still require source retrieval, C++ build compatibility, and repository-specific wrappers that return the same rigid transform and metric fields as the rest of the benchmark.

## Current thesis wording

Use language like this:

> We first evaluate a non-learning registration panel spanning identity, local ICP variants, robust ICP variants, feature-based global registration, probabilistic registration, generalized ICP, and external robust estimators. This establishes a deterministic baseline before introducing learned registration components. TEASER++ and TurboReg are reported when their project-relative native builds are available; remaining external solvers such as Super4PCS and Go-ICP are kept as optional dependency-gated baselines.
