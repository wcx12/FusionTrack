# MPS-GAF Registration

This directory contains the runnable registration code extracted from the
MPS-GAF experiments.  The dataset is intentionally kept outside the repository.

## Files

- `mps_gaf_registration_core.py`: model, graph fusion, Sinkhorn matching, and weighted SVD.
- `mps_gaf_data_pipeline.py`: ModelNet40 HDF5 loader and grouped multi-source batching.
- `mps_gaf_run.py`: command-line entry point for inspection, training, and evaluation.
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

```bash
python mps_gaf_run.py \
  --mode train \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --output_dir runs/mps_gaf_crop \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1 \
  --epochs 400
```

## Evaluation

```bash
python mps_gaf_run.py \
  --mode eval \
  --dataset_path /path/to/modelnet40_ply_hdf5_2048 \
  --checkpoint runs/mps_gaf_crop/mps_gaf_latest.pt \
  --output_dir runs/mps_gaf_crop_eval \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1
```

The grouped data loader is required for both training and evaluation.  Do not
replace it with a plain `DataLoader(batch_size=...)`, because the model assumes
that `num_sources_per_ref` adjacent rows belong to the same reference group.
