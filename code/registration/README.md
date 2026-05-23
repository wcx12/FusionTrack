# MPS-GAF 配准代码

这个目录包含从 MPS-GAF 实验中整理出来的可运行配准代码。数据集不放进仓库，训练数据、模型权重、运行日志和实验输出都应保存在仓库外部或被 `.gitignore` 忽略。

## 文件说明

- `mps_gaf_registration_core.py`：模型主体、图融合、Sinkhorn 匹配、加权 SVD、learned SVD 内点加权，以及可选几何细化工具。
- `mps_gaf_data_pipeline.py`：ModelNet40 HDF5 数据读取，以及 grouped multi-source batching。
- `mps_gaf_run.py`：检查、训练和评价的命令行入口。
- `EXPERIMENT_RECORD.md`：实验记录、消融结果、推荐 checkpoint 和 RPM-Net 对比结果。
- `requirements.txt`：Python 依赖。

## 数据集结构

请下载 `modelnet40_ply_hdf5_2048`，并放在仓库外部，例如：

```text
datasets/modelnet40_ply_hdf5_2048/
  shape_names.txt
  train_files.txt
  test_files.txt
  ply_data_train*.h5
  ply_data_test*.h5
```

## 运行前检查

训练前建议先运行检查命令。它会确认每个 batch 都包含完整的 source group，并且同一组 source 共享同一个 reference shape。

```bash
python mps_gaf_run.py \
  --mode inspect \
  --dataset_path datasets/modelnet40_ply_hdf5_2048 \
  --noise_type crop \
  --num_sources_per_ref 10 \
  --groups_per_batch 1
```

## 训练

当前推荐的 learning-forward 配置使用 entropy-weighted matching 加 learned SVD inlier head。这个 head 会预测哪些 source points 应该参与最终的加权 SVD 位姿求解。

```bash
python mps_gaf_run.py \
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

如果要进行完整训练，请去掉 `--max_train_steps`，并按需要设置更完整的 validation protocol。

## 单批次冒烟测试

这个命令用于快速检查数据读取、grouped batching、模型 forward、loss、backward、一次 optimizer step、validation forward 和 checkpoint 写入，不会跑完整实验。

```bash
python mps_gaf_run.py \
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

## 评价

```bash
python mps_gaf_run.py \
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

如果要复现 learned SVD 相对 RPM-Net 原始结果的最强记录，可以在评价时打开轻量 point-to-plane refinement：

```bash
python mps_gaf_run.py \
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

在已记录的 source-2、crop-noise、20-batch validation protocol 下，当前 learned SVD checkpoint 的结果如下：

| 方法 | 细化方式 | 旋转误差均值 | 平移误差均值 | Pose50 |
| --- | --- | ---: | ---: | ---: |
| RPM-Net original pose-best | 无 | 10.3604 | 0.0987 | 15.2972 |
| Learned MPS-GAF | 无 | 12.1518 | 0.1505 | 19.6781 |
| Learned MPS-GAF | plane，5 步 | 5.5346 | 0.0939 | 10.2313 |
| Learned MPS-GAF | plane，20 步 | 4.3478 | 0.0742 | 8.0559 |

完整消融历史和远程 artifact 路径见 `EXPERIMENT_RECORD.md`。

## 注意事项

- 训练和评价都必须使用 grouped data loader。不要替换成普通的 `DataLoader(batch_size=...)`，因为模型假设相邻的 `num_sources_per_ref` 行属于同一个 reference group。
- 训练时，source/reference augmentation 会在每个 epoch 按 dataset epoch seed 重新生成。
- validation 和 test 数据保持确定性，保证不同 run 之间的指标可比较。
