# Non-learning Baseline Plan (Phase 1)

## Rule: path policy

- All paths in this benchmark workflow should be relative paths only.
- Command snippets should use repository-relative paths, for example `datasets/modelnet40_ply_hdf5_2048`, not drive-letter paths or root-prefixed filesystem paths.

## Current status

- We are currently focusing on non-learning baselines only.
- Trained MPS-GAF checkpoints are on another server and are intentionally out of scope for this phase.
- The implemented benchmark scripts reject absolute dataset/output paths.
- The main fair-comparison protocol is `source-2 / crop-noise / 20-batch validation`, matching the learned-model evaluation setup.

## Protocol-aligned run (phase 1.2)

Executed on the remote server:

- script: `code/registration/run_registration_benchmark.py`
- dataset: `datasets/modelnet40_ply_hdf5_2048`
- split: `test`
- noise: `crop`
- `num_sources_per_ref`: `2`
- `groups_per_batch`: `1`
- `max_eval_batches`: `20`
- `num_points`: `1024`
- `partial`: `0.7 0.7`
- `rot_mag`: `45`
- `trans_mag`: `0.5`
- output: `runs/mps_gaf_nonlearn_schema_source2_crop_eval20_full`

Local artifacts:

- `code/registration/non_learning_source2_crop_eval20_summary.json`
- `code/registration/non_learning_mps_gaf_eval_schema_summary.json`
- `code/registration/non_learning_comparison_schema_summary.json`
- `code/registration/non_learning_source2_crop_protocol_results.md`

Important limitation:

- The full remote data copy is available under the standard relative dataset path.
- The exact 20-batch protocol evaluates `40` pairs and has been completed for the available non-learning methods.

## Completed non-learning sweep (phase 1.1)

Executed on the remote server:

- script: `code/registration/run_non_learning_baseline_sweep.py`
- dataset: `datasets/modelnet40_ply_hdf5_2048`
- split: `train`
- cases: `base_crop`, `sparse_crop`, `dense_crop`, `points_512`, `jitter`, `clean`
- threshold grid: `success_rotation_deg in {5,10,15}` and `success_translation in {0.2,0.3,0.5}`
- output root: `runs/mps_gaf_nonlearn_threshold_sweep_v1`

Methods:

- `identity`
- `icp_point_to_point`
- `icp_point_to_plane`
- `icp_trimmed`
- `ransac_icp`
- `fpfh_ransac`
- `fpfh_fgr`
- `gicp`
- `cpd`
- `teaserpp`
- `super4pcs`
- `goicp`

Local consolidated artifacts:

- `code/registration/threshold_sweep_payload.json`
- `code/registration/threshold_sweep_summary.md`
- `code/registration/non_learning_baseline_results.md`
- `code/registration/non_learning_baseline_research.md`

## Main observations

- TEASER++, Super4PCS, and Go-ICP are configured as method stubs but skipped in the current environment because their external dependencies are unavailable.
- `open3d` and `pycpd` are installed on the remote server, so ICP, FPFH/FGR, GICP, and CPD baselines can run.
- The current test split copy is too small for stable statistics, so phase 1.1 uses the train split while keeping the command reproducible.
- The source-2/crop phase 1.2 run now uses the full test split and evaluates the intended `40` pairs for the 20-batch protocol.
- The primary non-learning output now includes `mps_gaf_eval_schema`, matching `mps_gaf_run.py --mode eval` field names for method-by-method comparison.
- At the relaxed `15 deg / 0.5` threshold, `icp_point_to_point` gives the best macro Chamfer among running methods (`0.398`) and the highest macro success rate (`0.250`).
- At the same threshold, `icp_trimmed` gives the lowest macro rotation error (`24.475 deg`) and a competitive success rate (`0.194`).
- `ransac_icp` has strong Chamfer but poor rotation and higher runtime, so it should be reported as a shape-overlap-oriented baseline rather than the main accuracy winner.
- Strict thresholds are often too harsh for this data/noise setup, so the report should include both strict and relaxed success criteria.

## Phase 1 scope

The first phase stays strictly classical:

1. Identity baseline
2. Point-to-point ICP
3. Point-to-plane ICP
4. Trimmed ICP
5. RANSAC + ICP
6. FPFH + RANSAC
7. FPFH + FGR
8. Generalized ICP
9. Rigid CPD

External-method stubs:

- TEASER++
- Super4PCS / 4PCS
- Go-ICP
- NDT is tracked as a phase-2 candidate, but not included in phase 1 because the current Python/Open3D stack does not provide a stable NDT registration backend.

These external methods should be reported as skipped dependency cases unless their runtimes are installed and the same metric schema can run them.

## Script surface

Implemented files:

- `code/registration/non_learning_baselines.py`
- `code/registration/run_registration_benchmark.py`
- `code/registration/run_registration_benchmark_suite.py`
- `code/registration/run_non_learning_baseline_sweep.py`

The benchmark reports:

- mean / RMSE rotation error in degrees
- mean / RMSE translation error
- mean / RMSE Chamfer distance
- mean runtime per pair
- success rate under configurable thresholds
- skip rate and failed pair counts

The sweep script supports:

- `--dataset_split`
- `--case_set protocol|robustness`
- `--success_rotation_degs`
- `--success_translations`
- `--reuse_existing`
- relative-path validation for dataset and output paths

## Reproducible commands

Run the source-2/crop validation protocol:

```bash
python code/registration/run_registration_benchmark.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --dataset_split test \
  --output_dir runs/mps_gaf_nonlearn_schema_source2_crop_eval20_full \
  --methods identity,icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac,fpfh_fgr,gicp,cpd,teaserpp,super4pcs,goicp \
  --noise_type crop \
  --num_points 1024 \
  --partial 0.7 0.7 \
  --rot_mag 45 \
  --trans_mag 0.5 \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_workers 0 \
  --max_eval_batches 20 \
  --icp_iterations 20 \
  --icp_trim_fraction 0.7 \
  --success_rotation_deg 15 \
  --success_translation 0.5
```

Run one robustness case:

```bash
python code/registration/run_registration_benchmark.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --dataset_split train \
  --output_dir runs/mps_gaf_nonlearn_baselines \
  --methods identity,icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac,fpfh_fgr,gicp,cpd \
  --noise_type crop \
  --num_points 1024 \
  --partial 0.7 0.7 \
  --rot_mag 30 \
  --trans_mag 0.3 \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_workers 0 \
  --max_eval_batches 100 \
  --icp_iterations 20 \
  --icp_trim_fraction 0.7 \
  --success_rotation_deg 15 \
  --success_translation 0.5
```

Run the full threshold sweep:

```bash
python code/registration/run_non_learning_baseline_sweep.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --dataset_split train \
  --output_root runs/mps_gaf_nonlearn_threshold_sweep_v1 \
  --methods identity,icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac,fpfh_fgr,gicp,cpd,teaserpp,super4pcs,goicp \
  --case_set robustness \
  --rot_mag 30 \
  --trans_mag 0.3 \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_workers 0 \
  --max_eval_batches 20 \
  --icp_iterations 20 \
  --icp_trim_fraction 0.7 \
  --success_rotation_degs 5,10,15 \
  --success_translations 0.2,0.3,0.5
```

Regenerate summaries without rerunning benchmark cases:

```bash
python code/registration/run_non_learning_baseline_sweep.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --dataset_split train \
  --output_root runs/mps_gaf_nonlearn_threshold_sweep_v1 \
  --methods identity,icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac,fpfh_fgr,gicp,cpd,teaserpp,super4pcs,goicp \
  --case_set robustness \
  --success_rotation_degs 5,10,15 \
  --success_translations 0.2,0.3,0.5 \
  --reuse_existing
```

## Remaining next actions

1. Put the full validation/test data copy on the current server, or rerun the non-learning baselines on the server that produced the learned-model validation results.
2. If external C++ dependencies become available, implement wrappers for TEASER++, Super4PCS, and Go-ICP and run the same source-2/crop protocol without changing the metric schema.
3. When trained model weights are available, add learned MPS-GAF results as phase 2 using the same source-2/crop protocol, then keep the threshold sweep as appendix evidence.

## References

- Open3D ICP and global registration tutorials: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html and https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
- TEASER++ official repository: https://github.com/MIT-SPARK/TEASER-plusplus
- Super4PCS paper: https://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/super4pcs.pdf
- Go-ICP paper: https://arxiv.org/abs/1605.03344
- CPD paper: https://arxiv.org/abs/0905.2635
- FGR paper page: https://vladlen.info/publications/fast-global-registration/
