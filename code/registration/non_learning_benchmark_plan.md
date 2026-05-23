# Non-learning Baseline Plan (Phase 1)

## Rule: path policy

- All paths in this benchmark workflow should be **relative paths** only.
- When command snippets need data or output paths, use a repository-relative path (for example `datasets/modelnet40_ply_hdf5_2048`) rather than `/...` or drive-letter paths.

## Current status

- No trained **MPS-GAF** weights are present in this repository (`.pt/.pth`).
- `mps_gaf_run.py` writes checkpoints as `mps_gaf_latest.pt` / `mps_gaf_best.pt` when training is run, so if you have trained the model locally, these are generated under your chosen `--output_dir`.

## Phase 1 scope: non-learning baselines only

We keep this phase strictly classical:

1. point-to-point ICP
2. point-to-plane ICP
3. Trimmed ICP
4. RANSAC + ICP
5. Identity (baseline floor)
6. FPFH + RANSAC (Open3D path)
7. FPFH + FGR (Open3D path)
8. Generalized ICP (Open3D path)
9. CPD (pycpd path)

External-method stubs (run only when dependencies are available):

- TEASER++
- Super4PCS / 4PCS
- Go-ICP

These are already implemented in:

- `code/registration/non_learning_baselines.py`
- `code/registration/run_registration_benchmark.py`

FPFH + RANSAC is now available through method name `fpfh_ransac` when Open3D is installed (`pip install open3d`).

The benchmark script now reports:

- mean / RMSE rotation error (degrees)
- mean / RMSE translation error
- mean Chamfer distance / RMSE
- mean runtime per pair
- success rate (using configurable thresholds)

The suite script (`code/registration/run_registration_benchmark_suite.py`) runs a standard set of non-learning ablations and writes:

- `runs/mps_gaf_nonlearn_suite/<case>/baseline_summary.json` for each case
- `runs/mps_gaf_nonlearn_suite/suite_summary.json` as an aggregate summary

### Success threshold flags

- `--success_rotation_deg` (default: `5.0`)
- `--success_translation` (default: `0.2`)

### Example command (once torch is available)

```bash
python code/registration/run_registration_benchmark.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/mps_gaf_nonlearn_baselines \
  --methods icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac \
  --noise_type crop \
  --num_points 1024 \
  --partial 0.7 0.7 \
  --rot_mag 45 \
  --trans_mag 0.5 \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_workers 0 \
  --max_eval_batches 100 \
  --icp_iterations 20 \
  --icp_trim_fraction 0.7 \
  --success_rotation_deg 5 \
  --success_translation 0.2 \
  --icp_point_max_angle_deg 10 \
  --icp_point_max_translation 0.2 \
  --fpfh_voxel_size 0.05 \
  --fpfh_normal_radius 0.1 \
  --fpfh_feature_radius 0.25 \
  --fpfh_normal_max_nn 30 \
  --fpfh_feature_max_nn 100 \
  --fpfh_max_correspondence_distance 0.075 \
  --fpfh_ransac_n 4 \
  --fpfh_ransac_max_iterations 100000
``` 

```bash
python code/registration/run_registration_benchmark_suite.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048
```

This writes:

- `runs/mps_gaf_nonlearn_baselines/baseline_summary.json`
- Aggregated metrics under `benchmark` and per-pair logs in `pair_results`.

## Research shortlist for state-of-the-art classical methods

For next phases (not yet implemented in code), add methods in this order:

### Global + outlier-robust
- TEASER++
- Super4PCS / 4PCS
- Go-ICP

### Feature + local-global pipelines
- FPFH + RANSAC (Open3D)
- Fast Global Registration

### Probabilistic / dense-model registration
- CPD
- GICP / NDT

## References (for thesis)

- Open3D registration tutorials (ICP and global registration):  
  https://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_registration.html  
  https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
 - TEASER++ (official):  
  https://github.com/MIT-SPARK/TEASER-plusplus
- Super4PCS:  
  https://nmellado.github.io/Super4PCS/  
  https://github.com/nmellado/Super4PCS
- Go-ICP:  
  https://arxiv.org/abs/1605.03344
- CPD:  
  https://arxiv.org/abs/0905.2635
 - Open3D FGR: https://www.open3d.org/docs/latest/tutorial/Basic/file_viewer.html#fpfh-feature

## Execution plan after this phase

1. Keep Phase 1 as the main ablation table.
2. If you need stronger baselines, add one at a time in this order:
   - TEASER++
   - FPFH + RANSAC + local ICP refinement
   - Super4PCS
3. Keep code changes minimal and compare with the same metrics JSON format.

## GPU / dependency note

- This phase is **CPU-ready** for the baseline code itself.
- For `run_registration_benchmark.py`, you still need a working `torch` install to build the ModelNet40 dataloader and tensor batch loop.
- On this local machine, `torch` import fails (DLL issue). Please run the benchmark on your server with a proper GPU/CPU PyTorch runtime when you open it.
