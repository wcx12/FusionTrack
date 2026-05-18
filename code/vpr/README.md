# TF-VPR Code

This folder contains the Visual Place Recognition implementation used by FusionTrack.

Source: https://github.com/ddfs430/TF-VPR
Imported commit: `c2c9b0d06842f0d8c89e91f8a451a2d5214739b5`
License: MIT, see `LICENSE`.

Only code and text metadata are included here. Datasets, pretrained weights, run logs, cached files, and upstream display images are intentionally excluded from the repository.

## Layout

- `eval.py`: main evaluation entry point.
- `parser.py`: command-line arguments.
- `datasets_ws.py`: dataset loading for database/query image folders.
- `test.py`: feature extraction, nearest-neighbor retrieval, recall calculation, and per-query JSON output.
- `model/`: TF-VPR backbone and aggregation modules.
- `UPSTREAM_README.md`: original upstream README for reference.
- `requirements.txt`: upstream Python environment specification.

## Dataset Format

The loader expects VisualGeoLocalization-style datasets:

```text
<eval_datasets_folder>/<dataset_name>/images/test/database/*.jpg
<eval_datasets_folder>/<dataset_name>/images/test/queries/*.jpg
```

Image names must contain UTM coordinates in the upstream format:

```text
.../@<utm_easting>@<utm_northing>@...@.jpg
```

The default dataset root is `/data/datasets/vpr`; override it with `--eval_datasets_folder`.

## Pretrained Weights

The default DINOv2 checkpoint path is:

```text
/data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth
```

Override it with `--foundation_model_path`. Pretrained weights should remain outside git.

## Example

Run from this folder so the local imports resolve correctly:

```bash
python eval.py \
  --eval_dataset_name pitts30k \
  --eval_datasets_folder /data/datasets/vpr \
  --foundation_model_path /data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth \
  --backbone dinov2 \
  --mode TF_VPR \
  --num_clusters 17
```

Outputs are written under `test/<save_dir>/<timestamp>/`.
