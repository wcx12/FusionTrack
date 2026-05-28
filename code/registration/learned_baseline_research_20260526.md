# Learned Registration Baseline Audit - 2026-05-26

This note tracks learned point-cloud registration baselines considered for the benchmark expansion. It separates methods that have comparable results from methods that are still blocked by data, dependency, or pretrained-weight access.

All project commands and records should use repository-relative filesystem paths.

## Runnable or Already Evaluated

| method | venue/status | repository | current status |
|---|---|---|---|
| PointRegGPT GeoTransformer | ECCV 2024 release package | `https://github.com/Chen-Suyi/PointRegGPT` | The release weights package is staged on the server. The GeoTransformer-16w and shorter 2w checkpoints run under a lightweight export path and have completed 3DMatch, 3DLoMatch, and ETH converted-schema summaries. ETH uses all 713 official pairs and remains a cross-dataset check rather than ETH-specific training. |
| GeoTransformer | CVPR 2022 | `https://github.com/qinzheng93/GeoTransformer` | Official 3DMatch, 3DLoMatch, and ModelNet weights are available; 3DMatch/3DLoMatch official reruns and converted-schema summaries are recorded. A cross-dataset ETH run using the 3DMatch checkpoint has also completed over all 713 official ETH pairs after converting ETH to the GeoTransformer data schema. |
| RPMNet | CVPR 2020 era common baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema. |
| DCP-DGCNN | ICCV 2019 common baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema. |
| PRNet-DGCNN | common partial registration baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema. |
| IDAM-GNN | common partial registration baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema. |
| PointNetLK | common learning baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema. |
| OMNet | common overlap-masking baseline | existing local runner | Converged ModelNet and 3DMatch checkpoints exist; evaluated on ModelNet, 3DMatch, 3DLoMatch, and ETH under the project schema, but remains unstable. |

## Newer or Stronger Candidates

| method | venue/status | repository | blocker |
|---|---|---|---|
| CAST | NeurIPS 2024 | `https://github.com/RenlangHuang/CAST` | Code is now present on the server and the official release exposes `ckpt.zip`. A resumed checkpoint download reached a partial file but remained slow/flaky. `pip install pytorch3d` has no wheel for the current stack, and runtime also needs MinkowskiEngine. CAST expects PREDATOR-style `.ply` fragments while the available 3DMatch cache is `.pth`. |
| BUFFER-X | ICCV 2025 highlight per project README | `https://github.com/MIT-SPARK/BUFFER-X` | Code is present on the server and the Python package can install under the current Torch 2.8 environment. Full inference remains blocked because `nvcc` is absent, so `pointnet2_ops` cannot be built; `torch_batch_svd` is also missing after the failed extension phase. Hugging Face model repo listing for `Hyungtae-Lim/BUFFER-X` hung until interrupted, and the full benchmark data bundle is about 130 GB, far larger than the current free data-disk space. |
| PARE-Net | ECCV 2024 | `https://github.com/yaorz97/PARENet` | Code is present. The official README says pretrained 3DMatch and KITTI models are released through Google Drive/Baidu-style mirrors. A direct `gdown` file-id retry against the official Google Drive weight id failed with `Network is unreachable`; no checkpoint was produced. Runtime also expects the project `pareconv` extension and a PyTorch 1.13/CUDA 11.6-era stack, while the current server has PyTorch 2.8. |
| RoITr | CVPR 2023 rotation-invariant transformer baseline | `https://github.com/haoyu94/RoITr` | Code, checkpoint, patched `pointops`, and `tensorboardX` are now present on the server. 3DMatch and 3DLoMatch inference plus official 2500-correspondence evaluation completed. |
| RegTR | transformer registration baseline | `https://github.com/yewzijian/RegTR` | Official 3DMatch and ModelNet checkpoints are present, but runtime requires MinkowskiEngine and PyTorch3D. Current Python/PyTorch stack and disk space make a clean compatible environment impractical in this pass. |
| JPCR | CVPR 2024 component plus Jittor PCR collection | `https://github.com/zhiyuanYU134/JPCR` | Code is present. Pretrained weights are hosted on Google Drive, and the runtime requires Jittor plus `vision3d`; both are absent from the current environment. This is not a near-term fair baseline unless a separate Python 3.8/Jittor environment and weights are prepared. |
| PREDATOR / OverlapPredator | CVPR 2021 Oral, widely used low-overlap baseline | `https://github.com/prs-eth/OverlapPredator` | Official code is staged and the PointRegGPT PREDATOR weight is present, but the official stack targets Python 3.8, PyTorch 1.7.1, and CUDA 11.2. A current-server compile probe fails because `numpy.distutils` is unavailable in Python 3.12/NumPy 2-era environments. |
| PointRegGPT CoFiNet/PREDATOR weights | ECCV 2024 release package | `https://github.com/Chen-Suyi/PointRegGPT` | Weights and a CoFiNet data symlink are staged, but only the GeoTransformer-compatible checkpoints have been run so far. CoFiNet's official stack pins Torch 1.8/CUDA 11.1; its wrappers fail in the current Python 3.12/Torch 2.8 server because `numpy.distutils` and `THC/THC.h` are unavailable. CoFiNet/PREDATOR need old-stack environments or source patches before they are comparable under the project schema. |
| CoFiNet / RoReg / GCL-KPConv | common 3DMatch learned baselines | official GitHub repositories | Remote `git ls-remote` attempts to several candidate repositories timed out from the server. These remain lower-priority than PARE-Net because they either duplicate already-covered GeoTransformer/RoITr-era territory or require additional old-stack rebuilds and external weights. |

## Dataset Status

| dataset | standard split/source | current status |
|---|---|---|
| ModelNet40 | existing package train/test split | Full learned evaluation and source-2/crop eval20 non-learning results recorded. |
| 3DMatch | GeoTransformer/PREDATOR standard train/val/test metadata | Learned, non-learning, GeoTransformer, RoITr, and PointRegGPT GeoTransformer 16w/2w converted-schema results recorded. |
| 3DLoMatch | standard low-overlap test split from the 3DMatch benchmark family | Learned, non-learning, GeoTransformer, RoITr, and PointRegGPT GeoTransformer 16w/2w converted-schema results recorded. |
| ETH | standard four-scene ETH laser benchmark with `gt.log` pairs | Downloaded and evaluated over 713 official pairs for the current core method set, plus GeoTransformer and PointRegGPT GeoTransformer 16w/2w 3DMatch-checkpoint cross-dataset runs. |
| KITTI | GeoTransformer standard metadata, sequences 00-05 train, 06-07 val, 08-10 test | Metadata is present, but point clouds are absent and the full odometry velodyne archive does not fit comfortably on the current data disk. |
