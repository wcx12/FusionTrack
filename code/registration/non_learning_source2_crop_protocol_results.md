# Source-2 Crop Protocol Non-learning Baseline Results

Date: 2026-05-23

Remote run: `runs/mps_gaf_nonlearn_schema_source2_crop_eval20_full`

## Protocol

This run aligns the non-learning baselines with the MPS-GAF validation protocol:

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
- success threshold: `15 deg / 0.5`

## Data Availability

The full ModelNet40 HDF5 data copy is now available on the remote server under the standard relative dataset path:

| split | available pairs under source-2/crop protocol |
|---|---:|
| `train` | 19680 |
| `val` | 4936 |
| `test` | 4936 |

This run evaluates the MPS-GAF 20-batch protocol, i.e. `40` test pairs with `num_sources_per_ref=2` and `groups_per_batch=1`.

## MPS-GAF Eval Schema

The primary non-learning artifact is now projected onto the same metric schema emitted by `mps_gaf_run.py --mode eval`:

- `rotation_error_deg_mean`
- `rotation_error_deg_rmse`
- `translation_error_mean`
- `translation_error_rmse`
- `chamfer_distance_mean`
- `num_pairs`

Local artifacts:

- `code/registration/non_learning_mps_gaf_eval_schema_summary.json`
- `code/registration/non_learning_comparison_schema_summary.json`
- `code/registration/non_learning_main_table.md`
- `code/registration/non_learning_main_table.csv`
- `code/registration/non_learning_source2_crop_eval20_summary.json`
- `code/registration/non_learning_protocol_run_manifest.json`
- `code/registration/non_learning_external_dependency_probe.json`

## Results

| method | pairs | rot/deg | trans | chamfer | pose50 | success | skip | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 40 | 39.377 | 0.463 | 0.240 | 62.505 | 0.025 | 0.000 | 0.0000 |
| `icp_point_to_point` | 40 | 24.868 | 0.244 | 0.024 | 37.083 | 0.475 | 0.000 | 0.0244 |
| `icp_point_to_plane` | 40 | 32.144 | 0.277 | 0.050 | 46.012 | 0.525 | 0.000 | 0.0207 |
| `icp_trimmed` | 40 | 27.303 | 0.274 | 0.046 | 41.015 | 0.300 | 0.000 | 0.0241 |
| `ransac_icp` | 40 | 45.844 | 0.369 | 0.037 | 64.269 | 0.225 | 0.000 | 0.6327 |
| `fpfh_ransac` | 40 | 102.452 | 0.424 | 0.278 | 123.659 | 0.025 | 0.000 | 0.1980 |
| `fpfh_fgr` | 40 | 80.801 | 0.410 | 0.246 | 101.314 | 0.050 | 0.000 | 0.0453 |
| `gicp` | 40 | 62.958 | 0.345 | 0.313 | 80.187 | 0.000 | 0.000 | 0.0399 |
| `cpd` | 40 | 39.226 | 0.247 | 0.020 | 51.552 | 0.250 | 0.000 | 1.1773 |
| `teaserpp` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `super4pcs` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |
| `goicp` | 40 | inf | inf | inf | inf | 0.000 | 1.000 | 0.0000 |

## Interpretation

- Under the full 20-batch source-2/crop run, `cpd` gives the lowest Chamfer distance, while `icp_point_to_plane` gives the highest relaxed success rate.
- `icp_point_to_point` is the strongest practical local baseline by the combined balance of rotation, translation, Chamfer, success rate, and runtime.
- `teaserpp`, `super4pcs`, and `goicp` are dependency/wrapper skips, not measured accuracy failures.

## Verification

The current run was verified with these checks:

- Local JSON schema check: `schema-manifest-ok 12 methods`
- Remote output check: `remote-artifacts-ok`
- Script syntax check: `ast-ok`
- Relative-path policy scan over the current non-learning artifacts and benchmark entry scripts: no drive-letter or root-prefixed paths detected.

## Dependency Probe

The remote environment currently reports:

| dependency | available |
|---|---:|
| `open3d` | true |
| `pycpd` | true |
| `teaserpp_python` | false |
| `teaserpp` | false |
| `super4pcs` | false |
| `goicp` | false |

Installing the missing methods is a separate native-build task. TEASER++, Super4PCS, and Go-ICP require C++ solver builds or non-standard wrappers, and the current benchmark code still contains placeholder stubs for these methods. Installing packages alone is not enough; the wrappers must also emit the same transform and metric schema.

Dependency probe evidence:

- `pip install --dry-run teaserpp-python`: no matching distribution
- `pip install --dry-run super4pcs`: no matching distribution
- `pip install --dry-run pygoicp`: no matching distribution
- no `teaserpp`, `Super4PCS`, or `GoICP` CLI is available on the remote path
- `cmake` and `ninja` are callable through Python modules, but not present as shell PATH commands
- Super4PCS remote shallow clone failed with a connection reset; local shallow clone did not complete within the probe window

## Reproducible Command Shape

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
  --success_rotation_deg 15 \
  --success_translation 0.5
```
