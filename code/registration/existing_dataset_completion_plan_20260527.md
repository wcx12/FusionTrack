# Existing Dataset Completion Plan - 2026-05-27

## Scope

This pass only fills gaps for already-used datasets and already-integrated
methods. It does not add new methods or new datasets.

Datasets:

- `modelnet40_source2_crop`
- `3dmatch`
- `3dlomatch`
- `eth`

## Skip Policy

Complex, slow, or dependency-heavy cells are allowed to remain empty. Empty
cells must be recorded explicitly as skipped/null rather than silently omitted.

Skip a cell when any of the following applies:

- It requires a new CUDA extension build and the server has no `nvcc`.
- It requires a substantially different old stack, for example Python 3.8 with
  Torch 1.7/1.8, while the current run server uses Python 3.12 and Torch 2.8.
- It already timed out on a full dataset under a reduced budget.
- It would require substantial method-specific adapter work outside the current
  schema.
- It is expected to be very slow with low value for the current comparison.

## High-Priority Runnable Gaps

| dataset | method group | action |
|---|---|---|
| modelnet40_source2_crop | existing learned baselines | export or rerun eval20 so learned and non-learning rows share the same validation protocol |
| modelnet40_source2_crop | GeoTransformer / PointRegGPT / RoITr | skip unless an already-working adapter is available; otherwise record null |
| eth | RoITr | attempt only if the existing RoITr runtime can be reused without new extension builds; otherwise record null |
| 3dmatch / 3dlomatch / eth | Super4PCS | attempt if binary is already available; otherwise record null |

## Explicit Null Cells

| dataset | method | reason |
|---|---|---|
| eth | CPD rigid | full ETH remained too slow even with a reduced iteration budget |
| all external datasets | Go-ICP | expected full-matrix runtime is too high for low incremental value |
| all datasets | PREDATOR / CoFiNet / BUFFER-X | requires CUDA extension builds or old-stack environments, outside current scope |
| all datasets | CAST / RegTR / PARE-Net / JPCR | new-method work is outside this completion pass |

## Reporting Rule

Final tables should distinguish:

- numeric result: completed in the shared comparison schema
- `null`: intentionally skipped under this plan
- `n/a`: metric cannot be computed from available outputs, for example missing point-level predictions for Chamfer

