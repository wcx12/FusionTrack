# MPS-GAF 配准模块说明

本目录保存论文系统中的点云配准实验代码。当前阶段包含两类内容：

- 学习式 MPS-GAF 配准模型：用于多源点云融合、Sinkhorn 匹配、加权 SVD/learned SVD 估计位姿。
- 非学习配准基线：用于在系统还未完全替换算法前，先给 registration 模块提供可运行、可比较、可视化的结果。

所有数据集、运行输出、日志和模型权重都不应提交到仓库。服务器运行时请使用相对路径，例如 `datasets/modelnet40_ply_hdf5_2048` 和 `runs/...`，不要把本机或服务器绝对路径写进命令或 manifest。

## 目录结构

- `mps_gaf_registration_core.py`：MPS-GAF 主体、图融合、Sinkhorn 匹配、加权 SVD、learned SVD inlier head 和可选几何细化工具。
- `mps_gaf_data_pipeline.py`：ModelNet40 HDF5 数据读取、裁剪/扰动增强、grouped multi-source batching。
- `mps_gaf_run.py`：inspect、train、eval 的命令行入口。
- `run_registration_benchmark.py`：单组非学习配准基线评测入口。
- `run_registration_benchmark_suite.py`：多 preset 评测入口，支持 protocol 和 robustness 两种 case set。
- `run_non_learning_baseline_sweep.py`：成功阈值网格扫描入口，用于比较不同旋转/平移成功标准下的鲁棒性。
- `run_dcp_baseline.py`：将 DCP、PRNet、IDAM、RPMNet、PointNetLK 等外部学习式基线接入同一 MPS-GAF 评测协议。
- `non_learning_baselines.py`：ICP、trimmed ICP、RANSAC+ICP、FPFH、GICP、CPD、TEASER++、Super4PCS、Go-ICP 等方法封装。
- `EXPERIMENT_RECORD.md`：历史训练、消融和远程实验记录。
- `requirements.txt`：Python 依赖。

## 数据集要求

请下载 `modelnet40_ply_hdf5_2048`，并放在仓库外部或被 `.gitignore` 覆盖的位置。推荐服务器相对路径：

```text
datasets/modelnet40_ply_hdf5_2048/
  shape_names.txt
  train_files.txt
  test_files.txt
  ply_data_train*.h5
  ply_data_test*.h5
```

## 非学习基线：主协议

主协议是当前推荐用于系统占位和可视化接入的 registration 结果。它使用 source-2、crop-noise、20 eval batches，并输出 `baseline_summary.json`、`mps_gaf_eval_schema_summary.json` 和 `comparison_schema_summary.json`。

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

TEASER++、Super4PCS、Go-ICP 是可选外部依赖。脚本只接受仓库相对路径，例如：

```bash
python code/registration/run_registration_benchmark.py \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/external_nonlearn_source2_crop_eval20 \
  --methods teaserpp,super4pcs,goicp \
  --super4pcs_binary external_src/Super4PCS/build/demos/Super4PCS/Super4PCS \
  --goicp_binary external_src/Go-ICP/build/GoICP \
  --max_eval_batches 20
```

如果传入 Linux 根目录路径、临时目录路径或 Windows 盘符路径这类绝对路径，脚本会直接报错，避免把服务器或本机绝对路径写进实验记录。

## 非学习基线：鲁棒性阈值扫描

该命令会运行多个旋转/平移成功阈值，并生成 `threshold_sweep_payload.json` 与 `threshold_sweep_summary.md`。

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

如果每个阈值目录下已经存在 `suite_summary.json`，可以只重新聚合报告：

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

## 接入 FusionTrack 可视化系统

非学习 benchmark 输出 `baseline_summary.json` 后，需要通过 `fusiontrack.registration_adapter.build_registration_experiment_bundle` 转换为 dashboard 可读的三类文件：

- `registration_scores/*.jsonl`：每个方法一份 score rows，包含 `score`、`rotation_error_deg`、`translation_error`、`chamfer_distance`、`runtime_sec`、`success`、`skipped` 和 `component_scores`。
- `registration_metrics/*.json`：每个方法一份聚合指标，包含成功率、失败数、平均误差、平均耗时等。
- `registration_artifacts/registration_experiment_manifest.json`：最终 dashboard 的 registration manifest。
- `registration_points`：每条 score row 可附带 source/reference/aligned 三组点云，用于最终网页的配准诊断预览。

这些文件接入后，前端页面的任务下拉会出现 `Registration`。该任务没有人工异常 label，页面会按 score rows 选择序列，并把高误差/失败样本作为配准风险展示。适配器会优先读取 benchmark row 中真实的 `registration_points.source/reference/aligned` 点云，并在 score row 的 `metadata.registration_point_source` 中标注来源；如果当前实验产物还没有真实点云字段，才会回退到由配准误差确定性生成的轻量预览点云。后续学习式 MPS-GAF 或外部学习式配准基线只要输出同名字段，就能直接进入 dashboard。

Registration 展示层和 Individual/Group 的视频展示层是分开的：Individual/Group 使用 VT-Tiny-MOT 的原始 RGB 背景帧做四画面对比，Registration 使用点云配准动态 canvas 展示 source、reference 和 aligned 三组点云。`batch_****` 这类配准样本没有原始视频背景，这是任务定义差异，不是网页资源缺失。

## 学习式 MPS-GAF：检查数据

训练前建议先运行 inspect，确认每个 batch 都包含完整 source group，且同组 source 共享同一个 reference shape。

```bash
python code/registration/mps_gaf_run.py \
  --mode inspect \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1
```

## 学习式 MPS-GAF：训练

当前推荐的 learning-forward 配置使用 entropy-weighted matching + learned SVD inlier head。

```bash
python code/registration/mps_gaf_run.py \
  --mode train \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
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

如果要完整训练，请移除 `--max_train_steps`，并按需要扩大 validation protocol。

## 外部学习式基线：DCP/PRNet/IDAM/RPMNet/PointNetLK

`run_dcp_baseline.py` 用来把常见学习式点云配准方法统一到本项目的 source-2、crop-noise、MPS-GAF schema 下。它支持 `train` 和 `eval` 两种模式，输出 `eval_summary.json` 与 `comparison_schema_summary.json`，便于和 MPS-GAF、非学习基线放进同一张主表。

外部代码建议放在被 `.gitignore` 覆盖的 `external_src/learned_baselines/` 下：

```text
external_src/learned_baselines/
  DCP/
  PRNet/
  IDAM/
  RPMNet/
  PointNetLK/
```

示例：评估 DCP PointNet 版本。

```bash
python code/registration/run_dcp_baseline.py \
  --mode eval \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/dcp_pointnet_source2_crop_eval20_eval \
  --checkpoint runs/dcp_pointnet_source2_crop_eval20/dcp_best.pt \
  --model_family dcp \
  --dcp_repo external_src/learned_baselines/DCP \
  --emb_nn pointnet \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --max_eval_batches 20
```

示例：评估 RPMNet 或 PointNetLK。

```bash
python code/registration/run_dcp_baseline.py \
  --mode eval \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/rpmnet_source2_crop_eval20_eval \
  --checkpoint runs/rpmnet_source2_crop_eval20/rpmnet_best.pt \
  --model_family rpmnet \
  --rpmnet_repo external_src/learned_baselines/RPMNet \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --max_eval_batches 20
```

```bash
python code/registration/run_dcp_baseline.py \
  --mode eval \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/pointnetlk_source2_crop_eval20_eval \
  --checkpoint runs/pointnetlk_source2_crop_eval20/pointnetlk_best.pt \
  --model_family pointnetlk \
  --pointnetlk_repo external_src/learned_baselines/PointNetLK \
  --emb_dims 256 \
  --n_iters 3 \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --max_eval_batches 20
```

该 runner 会检查 `dataset_path`、`output_dir`、`checkpoint` 和外部仓库路径是否为相对路径；不同外部仓库加载时会清理 `model`/`util` 模块缓存，避免连续跑多个算法时互相串模块。

checkpoint 中保存的模型结构参数会在续训或评估时自动回填，例如 `model_family`、`emb_nn`、`emb_dims`、`n_iters`、RPMNet 半径/邻居数等。这样可以避免评估命令忘记写训练时的结构参数导致模型加载错位。如果确实需要完全使用命令行参数覆盖 checkpoint 里的结构配置，可以加：

```bash
--no_checkpoint_model_args
```

长时间训练时可以启用早停：

```bash
--early_stop_patience 20 \
--early_stop_min_delta 0.001
```

其中 `early_stop_patience` 表示连续多少个 epoch 没有超过 `early_stop_min_delta` 的改进后停止训练；默认值 `0` 表示不启用早停。

## 学习式 MPS-GAF：快速冒烟测试

```bash
python code/registration/mps_gaf_run.py \
  --mode train \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --output_dir runs/mps_gaf_smoke \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1 \
  --epochs 1 \
  --max_train_steps 1 \
  --max_eval_batches 1 \
  --device cpu
```

## 学习式 MPS-GAF：评估

```bash
python code/registration/mps_gaf_run.py \
  --mode eval \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --checkpoint runs/mps_gaf_learned_svd/mps_gaf_best.pt \
  --output_dir runs/mps_gaf_learned_svd_eval \
  --noise_type crop \
  --num_sources_per_ref 2 \
  --groups_per_batch 1 \
  --num_eval_iter 5 \
  --svd_weight_mode learned_entropy
```

如需复现实验记录里 learned SVD 相对 RPM-Net 的强结果，可以在评估时开启 point-to-plane refinement。

```bash
python code/registration/mps_gaf_run.py \
  --mode eval \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
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

## 注意事项

- 训练和评估必须使用 grouped data loader。不要替换成普通 `DataLoader(batch_size=...)`，因为模型假设相邻 `num_sources_per_ref` 行属于同一个 reference group。
- 训练时 source/reference augmentation 会在每个 epoch 根据 dataset epoch seed 重新生成。
- validation 和 test 数据保持确定性，保证不同 run 的指标可比较。
- 服务器运行命令全部使用相对路径；如果脚本检测到 `dataset_path`、`output_dir`、`checkpoint` 或外部二进制/外部仓库路径是绝对路径，会直接报错。
