# Existing Dataset Completion Matrix - 2026-05-27

This record is limited to existing datasets and already-integrated methods. New
methods and new datasets are out of scope for this pass.

Status labels:

- `done`: numeric result already exists in the shared comparison schema or report.
- `partial`: comparable RRE/RTE/pose exists, but one field such as Chamfer is unavailable.
- `null`: intentionally left empty under the skip policy.
- `todo`: runnable and still worth filling.

## Server Check

The new run server is usable for non-extension workloads:

- GPU: 48 GB vGPU
- Data disk: 50 GB
- Python: 3.12
- Torch: 2.8 CUDA 12.8
- `nvcc`: unavailable
- Server registration tests: 36 passed after installing `scipy`, `h5py`, `open3d`, `pycpd`, and `pytest`
- Latest local registration verification: 41 passed

Because `nvcc` is unavailable, cells requiring CUDA extension builds remain
`null`.

## Matrix

| method | ModelNet40 source-2/crop | 3DMatch | 3DLoMatch | ETH |
|---|---|---|---|---|
| MPS-GAF | done full-set; done eval20 short retrain smoke | done | done | done |
| RPMNet | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| DCP-DGCNN | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| PointNetLK | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| PRNet-DGCNN | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| IDAM-GNN | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| OMNet | done full-set; null eval20 alignment because checkpoint/external repo is unavailable locally | done | done | done |
| GeoTransformer | partial official protocol only; null for source-2/crop adapter | done | done | done cross-dataset |
| RoITr | null for source-2/crop adapter | partial | partial | null unless existing RoITr runtime/data can be reused |
| PointRegGPT GeoTransformer-16w | null for source-2/crop adapter | done | done | done cross-dataset |
| PointRegGPT GeoTransformer-2w | null for source-2/crop adapter | done | done | done cross-dataset |
| Identity | done eval20 | done | done | done |
| ICP point-to-point | done eval20 | done | done | done |
| ICP point-to-plane | done eval20 | done | done | done |
| Trimmed ICP | done eval20 | done | done | done |
| GICP | done eval20 | done | done | done |
| RANSAC-ICP | done eval20 | done | done | done |
| FPFH-FGR | done eval20 | done | done | done |
| FPFH-RANSAC | done eval20 | done | done | done with 5k cap |
| CPD rigid | done eval20 | done | done | null; ETH full run timed out under reduced budget |
| TurboReg | done eval20 | done | done | done |
| TEASER++ | done eval20 | done | done | done |
| MAC-FPFH | done eval20 | done | done | null; ETH data is not staged on the current server |
| SC2-PCR-FPFH | done eval20 | done | done | null; ETH data is not staged on the current server |
| KISS-Matcher | done eval20 | done | done | null; ETH data is not staged on the current server |
| Super4PCS | done eval20 | null unless binary is already available | null unless binary is already available | null unless binary is already available |
| Go-ICP | null; too slow for current matrix | null; too slow for current matrix | null; too slow for current matrix | null; too slow for current matrix |

## Immediate Runnable Work

The only high-value runnable cell left on the new server was ModelNet40 eval20
alignment for existing learned checkpoints. Local synchronized registration
checkpoints were not found.

A short MPS-GAF retrain was run as a server/path smoke check, not as a converged
replacement for the full-set ModelNet result:

- output: `code/registration/server_artifacts/20260526_new_baselines/runs/mps_gaf_modelnet_eval20_retrain_eval`
- checkpoint source: 3-epoch short retrain on the new server
- eval scope: 20 batches with `groups_per_batch=2`, 80 pairs
- pose: 37.619
- RRE: 23.293
- RTE: 0.287
- Chamfer: 0.027

The remaining learned eval20 alignment cells are `null` under the skip policy
because checkpoints and external repos were unavailable locally, and retraining
all models from scratch is outside the current completion pass.
