# MPS-GAF Registration

This directory contains the runnable registration code extracted from the
MPS-GAF experiments.  The dataset is intentionally kept outside the repository.

## Files

- `mps_gaf_registration_core.py`: model, graph fusion, Sinkhorn matching,
  weighted SVD, learned SVD inlier weighting, and optional geometric refinement
  utilities.
- `mps_gaf_data_pipeline.py`: ModelNet40 HDF5 loader and grouped multi-source batching.
- `mps_gaf_run.py`: command-line entry point for inspection, training, and evaluation.
- `EXPERIMENT_RECORD.md`: experiment history, ablations, recommended checkpoints,
  and RPM-Net comparison results.
- `requirements.txt`: Python dependencies.

## Dataset Layout

Download and keep `modelnet40_ply_hdf5_2048` outside this repository, for example:

```text
datasets/modelnet40_ply_hdf5_2048/
  shape_names.txt
  train_files.txt
  test_files.txt
  ply_data_train*.h5
  ply_data_test*.h5
```

## Sanity Check

Run this before training.  It verifies that each batch contains complete
source groups sharing the same reference shape.

```bash
python mps_gaf_run.py \
  --mode inspect \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1
```

## Training

The current recommended learning-forward configuration uses entropy-weighted
matching plus a learned SVD inlier head.  The head predicts which source points
should influence the final weighted SVD pose solve.

```bash
python mps_gaf_run.py \
  --mode train \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --output_dir runs/mps_gaf_learned_svd \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --epochs 180 \
  --max_train_steps 10 \
  --max_eval_batches 20 \
  --lr 5e-5 \
  --num_train_iter 2 \
  --num_eval_iter 5 \
  --fusion_mode full \
  --fusion_start_iter 1 \
  --no_self_corr \
  --fusion_logit_init -8 \
  --svd_weight_mode learned_entropy \
  --wt_chamfer 0.10 \
  --wt_transform 0.10 \
  --wt_svd_inlier 0.05 \
  --svd_inlier_radius 0.07 \
  --best_metric pose \
  --pose_trans_weight 50
```

For full training without a smoke-test cap, remove `--max_train_steps` and set a
larger validation protocol as needed.

## One-Batch Smoke Test

This checks data loading, grouped batching, model forward, loss, backward, one
optimizer step, validation forward, and checkpoint writing without running a
full experiment.

```bash
python mps_gaf_run.py \
  --mode train \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --output_dir runs/mps_gaf_smoke \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1 \
  --epochs 1 \
  --max_train_steps 1 \
  --max_eval_batches 1 \
  --device cpu
```

## Evaluation

```bash
python mps_gaf_run.py \
  --mode eval \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --checkpoint runs/mps_gaf_learned_svd/mps_gaf_best.pt \
  --output_dir runs/mps_gaf_learned_svd_eval \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_eval_iter 5 \
  --svd_weight_mode learned_entropy
```

To reproduce the strongest learned-SVD result against the RPM-Net original
output, enable light point-to-plane refinement:

```bash
python mps_gaf_run.py \
  --mode eval \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --checkpoint runs/mps_gaf_learned_svd/mps_gaf_best.pt \
  --output_dir runs/mps_gaf_learned_svd_plane20 \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --max_eval_batches 20 \
  --num_eval_iter 5 \
  --fusion_mode full \
  --fusion_start_iter 1 \
  --no_self_corr \
  --fusion_logit_init -8 \
  --svd_weight_mode learned_entropy \
  --icp_mode plane \
  --icp_refine_steps 20 \
  --icp_trim_fraction 0.7 \
  --icp_max_angle_deg 10 \
  --icp_max_translation 0.2
```

Under the recorded source-2, crop-noise, 20-batch validation protocol, the
current learned-SVD checkpoint achieved:

| Method | Refinement | Rot mean | Trans mean | Pose50 |
|---|---|---:|---:|---:|
| RPM-Net original pose-best | none | 10.3604 | 0.0987 | 15.2972 |
| Learned MPS-GAF | none | 12.1518 | 0.1505 | 19.6781 |
| Learned MPS-GAF | plane, 5 steps | 5.5346 | 0.0939 | 10.2313 |
| Learned MPS-GAF | plane, 20 steps | 4.3478 | 0.0742 | 8.0559 |

See `EXPERIMENT_RECORD.md` for the complete ablation history and exact remote
artifact paths.

The grouped data loader is required for both training and evaluation.  Do not
replace it with a plain `DataLoader(batch_size=...)`, because the model assumes
that `num_sources_per_ref` adjacent rows belong to the same reference group.

During training, source/reference augmentations are regenerated every epoch by
updating the dataset epoch seed.  Validation and test datasets remain
deterministic so metrics are comparable across runs.
