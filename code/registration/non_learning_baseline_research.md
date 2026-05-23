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
| External robust/global | TEASER++, Super4PCS, Go-ICP | stubbed; skipped without deps | Strong classical references, but need external C++/binding setup. |
| Voxel distribution | NDT | phase-2 candidate | Widely used in robotics/SLAM, but not included in phase 1 because the current Python/Open3D stack has no stable NDT registration implementation. |

## Literature-backed method map

- ICP remains the canonical local baseline. Use point-to-point and point-to-plane variants as separate rows because they fail differently under normals, partial overlap, and poor initialization. Open3D documents both point-to-point and point-to-plane ICP registration pipelines: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
- FPFH + RANSAC is the standard feature-based coarse registration baseline. FPFH was introduced for 3D registration in ICRA 2009, and Open3D uses FPFH features in its global registration tutorial: https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf and https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
- Fast Global Registration (FGR) is a strong classical global method for partially overlapping surfaces, published at ECCV 2016. It optimizes a robust objective over candidate matches without inner-loop closest-point updates: https://vladlen.info/publications/fast-global-registration/
- TEASER++ is a key robust, certifiable registration reference from RSS/T-RO era work. It is especially relevant when correspondences contain many outliers, but its Python/C++ dependency path should stay optional in this repository: https://github.com/MIT-SPARK/TEASER-plusplus
- Super4PCS is a classic global alignment method for arbitrary initial poses and low-overlap cases. It is a useful external dependency candidate after the Open3D/PyCPD baselines are stable: https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf
- Go-ICP is a globally optimal branch-and-bound ICP formulation. It is important as a classical reference, but usually slower and dependency-heavy, so it is better treated as optional external validation: https://arxiv.org/abs/1605.03344
- CPD models registration probabilistically and supports rigid and non-rigid settings. The current phase only uses rigid CPD: https://arxiv.org/abs/0905.2635
- NDT is a widely used classical registration family in robotics and mapping, but it usually comes through PCL, Autoware, or SLAM-specific stacks rather than the current Python-only benchmark stack. Keep it as a phase-2 dependency candidate instead of adding a partial reimplementation.
- Recent surveys show the field's top-venue frontier has moved heavily toward deep registration, but traditional methods are still necessary for reproducible non-learning baselines and ablation floors. A 2024 Pattern Recognition survey covers rigid pairwise registration broadly, while IJCAI 2024 focuses on deep-learning point-cloud registration taxonomy: https://www.sciencedirect.com/science/article/pii/S0031320324001596 and https://www.ijcai.org/proceedings/2024/922
- KISS-Matcher is a recent open-source classical/geometry-heavy direction that revisits FPFH-style matching with graph pruning. It is worth tracking for phase 2, but not mixing into phase 1 until external C++ dependencies are controlled: https://arxiv.org/abs/2409.15615

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

Keep NDT out of the first table unless a stable dependency is installed and it can emit the same metrics without changing the benchmark protocol.

Report TEASER++, Super4PCS, and Go-ICP as "configured but dependency unavailable" unless their external builds are installed and the same script can execute them. Do not compare missing external methods as failed accuracy results; report them as skipped dependency cases.

## Current dependency status

The remote environment currently has the Python dependencies needed for the running phase-1 baselines:

- `open3d`: available
- `pycpd`: available

The following external methods are not available yet:

- `teaserpp_python` / `teaserpp`
- `super4pcs`
- `goicp`

These are not simple missing imports in the same sense as a pure Python package. A remote probe found no matching pip distributions for `teaserpp-python`, `super4pcs`, or `pygoicp`, no existing solver CLI binaries, and no solver build that produced an executable during the probe. CMake/Ninja are callable through Python modules, but the external solvers still require source retrieval, C++ build compatibility, and repository-specific wrappers that return the same rigid transform and metric fields as the rest of the benchmark. On the current Windows-style remote environment, they should be installed and verified one at a time rather than added as an unpinned bulk dependency.

## Current thesis wording

Use language like this:

> We first evaluate a non-learning registration panel spanning identity, local ICP variants, robust ICP variants, feature-based global registration, probabilistic registration, and generalized ICP. This establishes a deterministic baseline before introducing learned registration components. Methods requiring external C++ solvers, such as TEASER++, Super4PCS, and Go-ICP, are included in the benchmark interface but reported separately when their runtime dependencies are unavailable.
