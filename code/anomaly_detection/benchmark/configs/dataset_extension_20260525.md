# 数据集扩展记录 2026-05-25

## 范围

这份记录说明 FusionTrack 异常检测 benchmark 的第一阶段数据集扩展。目标是让系统不只依赖 VT-Tiny-MOT，还可以接入更通用的多目标跟踪数据：

1. M3OT 风格的 RGB/IR 多目标跟踪数据。
2. MOT 系列 RGB 跟踪数据：MOT17、MOT20、DanceTrack、SportsMOT。

扩展后的统一入口仍然是已有中间文件：

```text
observations_<split>.csv
```

下游流程保持不变：

```text
observations_<split>.csv
  -> individual_trajectories_<split>.jsonl
  -> group_windows_<split>.jsonl
  -> benchmark matrix
```

也就是说，新 adapter 只负责把不同数据集规范化成 FusionTrack 已有格式，不改变后面的异常注入、个体级评测、群体级评测和 strict key audit。

## 已实现内容

新增 runner：

```text
code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py
code/anomaly_detection/benchmark/runners/prepare_tracking_dataset_protocol.py
```

新增 adapter 模块：

```text
code/anomaly_detection/benchmark/dataset_adapters/tracking_observations.py
```

支持的数据 profile：

| Profile | 目标数据集 | 原始输入 | 输出模态 |
| --- | --- | --- | --- |
| `motchallenge` | MOT17、MOT20 | `sequence/gt/gt.txt` | RGB |
| `dancetrack` | DanceTrack | `sequence/gt/gt.txt` | RGB |
| `sportsmot` | SportsMOT | `sequence/gt/gt.txt` | RGB |
| `m3ot` | M3OT RGB/IR 双根目录 | RGB 与 IR 各自的 `sequence/gt/gt.txt` | RGB + thermal |

## 原始格式映射

adapter 至少要求 MOT 风格标注的前 7 列：

```text
frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence_or_valid
```

如果存在第 8、9 列，则解释为：

```text
category_id, visibility
```

DanceTrack 和 SportsMOT 中如果类别信息没有实际区分意义，默认按一个前景类别处理。

## 标准输出字段

生成的 CSV 保留既有 trajectory 和 group-window 导出器需要的字段：

```text
dataset, sequence, track_id, category_id, category_name, fps, frame_id
rgb_file, rgb_x, rgb_y, rgb_w, rgb_h, rgb_cx, rgb_cy, ...
thermal_file, thermal_x, thermal_y, thermal_w, thermal_h, thermal_cx, thermal_cy, ...
modal_offset_dx_thermal_minus_rgb, modal_offset_dy_thermal_minus_rgb,
modal_offset_distance, modal_bbox_iou
```

MOT 系列 RGB-only 数据集会保留 thermal 和 modal relation 字段为空。M3OT paired conversion 会同时填充 RGB 和 thermal 分支，并计算跨模态中心偏移和 bbox IoU。

## 通用协议生成 runner

完成格式转换后，使用数据集通用 protocol runner，而不是 VT-Tiny-MOT 专用 extractor：

```bash
python code/anomaly_detection/benchmark/runners/prepare_tracking_dataset_protocol.py \
  --dataset MOT17 \
  --mode validation \
  --observations-csv /work/observations_train.csv \
  --output-root /work/fusiontrack_mot17_protocol \
  --window-size 16 \
  --stride 8 \
  --seed 42
```

真正 holdout 实验示例：

```bash
python code/anomaly_detection/benchmark/runners/prepare_tracking_dataset_protocol.py \
  --dataset M3OT \
  --mode holdout \
  --train-observations-csv /work/m3ot_train_observations.csv \
  --eval-observations-csv /work/m3ot_test_observations.csv \
  --output-root /work/fusiontrack_m3ot_holdout_protocol \
  --split-name test \
  --window-size 16 \
  --stride 8 \
  --seed 42
```

这个 runner 会写出和 VT-Tiny-MOT protocol 一致的下游文件：

```text
fused_trajectories_train.jsonl
fused_trajectories_<split>_clean.jsonl
fused_trajectories_<split>.jsonl
individual_labels_<split>.jsonl
group_windows_train.jsonl
group_windows_<split>_clean.jsonl
group_windows_<split>.jsonl
group_labels_<split>.jsonl
individual_<split>_matrix.json
group_<split>_matrix.json
protocol_manifest.json
```

## 转换命令示例

MOT17/MOT20：

```bash
python code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py \
  --dataset MOT17 \
  --profile motchallenge \
  --mot-root /data/MOT17/train \
  --output-csv /work/observations_train.csv \
  --summary-json /work/observations_train_summary.json
```

DanceTrack：

```bash
python code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py \
  --dataset DanceTrack \
  --profile dancetrack \
  --mot-root /data/DanceTrack/train \
  --output-csv /work/observations_train.csv \
  --summary-json /work/observations_train_summary.json
```

SportsMOT：

```bash
python code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py \
  --dataset SportsMOT \
  --profile sportsmot \
  --mot-root /data/SportsMOT/train \
  --output-csv /work/observations_train.csv \
  --summary-json /work/observations_train_summary.json
```

M3OT paired RGB/IR：

```bash
python code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py \
  --dataset M3OT \
  --profile m3ot \
  --rgb-root /data/M3OT/RGB/train \
  --thermal-root /data/M3OT/IR/train \
  --output-csv /work/observations_train.csv \
  --summary-json /work/observations_train_summary.json
```

## 关键参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--keep-category-id` | 支持 profile 默认保留类别 `1` | 保留指定类别。需要多个类别时可重复传入。 |
| `--include-all-categories` | 关闭 | 关闭默认类别过滤。只建议在明确做多类别实验时使用。 |
| `--frame-digits` | DanceTrack 为 `8`，其他 profile 为 `6` | 控制 `rgb_file` / `thermal_file` 中帧号文件名宽度。 |
| `--include-ignored` | 关闭 | 保留 MOT confidence/valid flag `<= 0` 的行。主实验通常不要打开。 |

## 实验治理规则

adapter 只完成数据规范化。它本身不代表某个方法结果已经可以进入论文主表。

跨数据集汇报结果前必须满足：

1. 同一数据集、同一 split 下，所有方法使用同一异常注入协议和同一 strict key audit。
2. 超参数只能在 validation 上选择。
3. 最终泛化结论需要在 held-out split 上确认。
4. 深度官方 baseline 必须记录 `loss_history.json`、`best_epoch`、`final_epoch`、early-stop reason、GPU 名称、wall time 和收敛状态。
5. 非学习方法标记为 `no-epoch`，不能写成 converged。
6. 可以提交转换摘要和 benchmark manifest，但不要提交原始数据集、checkpoint、大型 score 文件或服务器归档。

## 当前阶段结论

当前阶段已经完成：

```text
M3OT/MOT-family adapter implemented
dataset-generic protocol runner implemented
unit fixtures implemented
large external datasets not downloaded in repository
full cross-dataset experiments not yet run
```

本地验证：

```text
test_tracking_observation_adapters.py: 3 passed
test_prepare_tracking_dataset_protocol.py: 1 passed
```

2026-05-25 服务器阶段记录已经补充到：

```text
code/anomaly_detection/benchmark/configs/dataset_extension_server_results_20260525.md
```

该阶段完成了 MOT17 与 SportsMOT 的通用协议、标准矩阵、official-source GPU baseline 收敛补跑和 strict key audit。M3OT 仍因官方 Figshare 访问返回 HTTP 403，不能声明完成。
