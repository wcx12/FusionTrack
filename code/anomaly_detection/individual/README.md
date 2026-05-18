# MTF-BA

MTF-BA 是一套面向 **VT-Tiny-MOT** 数据集的多模态轨迹异常检测实验流水线。当前已经实现的是“单目标/个体轨迹异常检测”部分：从 RGB/Thermal 双模态标注中抽取轨迹，构造单目标轨迹样本，导出 6 路单模态特征，训练 LSTM Autoencoder 检测器，生成异常分数，并做单模态检测器集成。

后续“群体异常检测”部分还没有实现具体模型，但本仓库已经预留了稳定接口：`mtf_ba.group_interface` 和 `export_vt_tiny_mot_group_windows.py` 会把同一份观测 CSV 转成 scene/window 级输入样本，后续 group 模型只需要消费这些窗口并输出 object-aligned 的 group 分数即可接入融合层。

## 当前能力

已完成：

- 从 VT-Tiny-MOT COCO 风格标注中提取 RGB/Thermal 配对轨迹。
- 导出 object-centric 单目标轨迹 JSONL。
- 导出 6 组单模态特征：
  - `route_rgb`
  - `speed_rgb`
  - `shape_rgb`
  - `route_thermal`
  - `speed_thermal`
  - `shape_thermal`
- 按 sequence 切分 train/val，避免同一场景泄漏。
- 为每组特征训练一个 LSTM Autoencoder 检测器。
- 用重构误差和 embedding 邻域差异生成 detector-level anomaly score。
- 对 6 路分数做 mean/max rank ensemble，并保留 complementary elimination 分析结果。
- 渲染 sequence-level anomaly heatmap 的脚本。
- 预留群体异常检测输入窗口和分数输出接口。

尚未完成：

- 群体异常检测模型本体。
- 个体分数与群体分数的最终融合训练/评估脚本。
- 统一的 `requirements.txt` 或 `pyproject.toml`。

## 目录结构

```text
MTF-BA/
├── mtf_ba/
│   ├── schemas.py                    # 共享 ID 和分数记录结构
│   ├── individual_trajectories.py    # 单目标轨迹视图
│   ├── single_modality_features.py   # route/speed/shape 特征构造
│   ├── feature_training.py           # LSTM Autoencoder 训练逻辑
│   ├── feature_scoring.py            # 单检测器打分逻辑
│   ├── ensemble_scoring.py           # 6 路检测器分数集成
│   ├── group_interface.py            # 预留的群体异常检测接口
│   └── trajectory_jsonl.py            # JSONL 读取工具
├── dataset/
│   └── VT-Tiny-MOT/                  # 默认数据集目录
├── outputs/                          # 已生成的中间结果、模型、分数和可视化
├── docs/
│   ├── anomaly_pipeline_plan.md       # 分层设计计划
│   └── 运行操作文档.md                # 更细的中文运行说明
├── extract_vt_tiny_mot_trajectories.py
├── export_vt_tiny_mot_individual_trajectories.py
├── export_single_modality_features.py
├── split_train_val_by_sequence.py
├── train_all_single_modality_detectors.py
├── score_all_single_modality_detectors.py
├── run_single_modality_ensemble.py
├── analyze_single_modality_results.py
├── render_sequence_anomaly_heatmaps.py
└── export_vt_tiny_mot_group_windows.py
```

## FusionTrack 仓库中的数据位置

本目录只提交可运行代码、文档和依赖说明，不提交数据集、模型权重或输出文件。在 FusionTrack 仓库中，建议把 VT-Tiny-MOT 数据集放在：

```text
code/anomaly_detection/datasets/VT-Tiny-MOT/
```

从本目录运行脚本时，默认数据路径已经调整为：

```text
../datasets/VT-Tiny-MOT
```

如果数据集放在其它位置，可以在提取和渲染阶段显式传入 `--data-root`。

## 环境准备

建议使用 Python 3.10 或 3.11。当前仓库还没有锁定依赖文件，按代码导入至少需要：

- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `scipy`
- `tqdm`
- `kneed`
- `matplotlib`

Windows PowerShell 示例：

```powershell
cd D:\HuaweiMoveData\Users\Cory\Desktop\MTF-BA
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn scipy tqdm kneed matplotlib
pip install torch
```

如果要用 GPU，请按本机 CUDA 版本安装对应的 PyTorch。安装后可以检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 数据要求

当前主流水线在 FusionTrack 仓库中默认读取：

```text
../datasets/VT-Tiny-MOT/
├── annotations/
│   ├── instances_00_train2017.json
│   ├── instances_01_train2017.json
│   ├── instances_00_test2017.json
│   └── instances_01_test2017.json
├── train2017/
└── test2017/
```

其中：

- `00` 表示 RGB 模态。
- `01` 表示 Thermal 模态。
- 每个 sequence 目录下通常包含 `00/`、`01/` 和 `seqinfo.ini`。
- 如果数据路径不同，在提取阶段传入 `--data-root`。

仓库中还存在 `dataset/VISO/`，但当前这套 VT-Tiny-MOT 主流水线默认不从该目录读取。

## 标准运行流程

下面是从原始标注到个体异常分数的完整流程。

### 1. 提取双模态轨迹观测

```powershell
python extract_vt_tiny_mot_trajectories.py --split train
python extract_vt_tiny_mot_trajectories.py --split test
```

默认输出：

```text
outputs/vt_tiny_mot_trajectories/
├── observations_train.csv
├── observations_test.csv
├── trajectories_train.jsonl
├── trajectories_test.jsonl
├── summary_train.json
└── summary_test.json
```

`observations_<split>.csv` 是后续个体轨迹、群体窗口、热力图等模块共享的数据入口。

### 2. 导出单目标轨迹

```powershell
python export_vt_tiny_mot_individual_trajectories.py `
  --split train `
  --csv-path outputs/vt_tiny_mot_trajectories/observations_train.csv

python export_vt_tiny_mot_individual_trajectories.py `
  --split test `
  --csv-path outputs/vt_tiny_mot_trajectories/observations_test.csv
```

默认输出：

```text
outputs/vt_tiny_mot_individual/
├── individual_trajectories_train.jsonl
├── individual_trajectories_test.jsonl
├── individual_trajectories_summary_train.json
└── individual_trajectories_summary_test.json
```

单目标轨迹的核心 ID 约定：

```text
sample_id = "{sequence}:{track_id}"
```

这个 ID 会贯穿特征、训练、打分、集成和后续 group/fusion。

### 3. 导出单模态特征

```powershell
python export_single_modality_features.py `
  --jsonl-path outputs/vt_tiny_mot_individual/individual_trajectories_train.jsonl `
  --split train

python export_single_modality_features.py `
  --jsonl-path outputs/vt_tiny_mot_individual/individual_trajectories_test.jsonl `
  --split test
```

默认输出：

```text
outputs/vt_tiny_mot_features/
├── route_rgb_train.pkl
├── speed_rgb_train.pkl
├── shape_rgb_train.pkl
├── route_thermal_train.pkl
├── speed_thermal_train.pkl
├── shape_thermal_train.pkl
└── *_test.pkl
```

特征含义：

- `route_*`：按累计路程重采样后的相对中心轨迹。
- `speed_*`：逐帧中心点速度。
- `shape_*`：去重、归一化、重采样并 PCA 后的运动形状特征。

### 4. 按 sequence 切分 train/val

```powershell
python split_train_val_by_sequence.py --split-features
```

默认输出：

```text
outputs/vt_tiny_mot_individual_split/
├── individual_trajectories_train.jsonl
├── individual_trajectories_val.jsonl
└── train_val_split_summary.json

outputs/vt_tiny_mot_features_split/
├── route_rgb_train.pkl
├── route_rgb_val.pkl
└── ...
```

这里按 sequence 切分，而不是按 `sample_id` 随机切分，目的是避免同一场景中的对象同时进入训练和验证。

### 5. 训练 6 个单模态检测器

```powershell
python train_all_single_modality_detectors.py
```

常用参数：

```powershell
python train_all_single_modality_detectors.py `
  --hidden-size 128 `
  --batch-size 64 `
  --learning-rate 1e-4 `
  --num-epochs 100 `
  --early-stopping-patience 20 `
  --cuda-device cuda:0
```

默认输出：

```text
outputs/vt_tiny_mot_models/
├── route_rgb/
│   ├── best_model.pth
│   ├── normalization_stats.json
│   └── train_summary.json
└── ...
```

如果没有可用 GPU，代码会自动回退到 CPU。

### 6. 对检测器打分

```powershell
python score_all_single_modality_detectors.py --split test
```

如果需要 train/val 分数：

```powershell
python score_all_single_modality_detectors.py --split train
python score_all_single_modality_detectors.py --split val
```

默认输出：

```text
outputs/vt_tiny_mot_scores/
├── route_rgb/
│   └── test/
│       ├── embeddings.pkl
│       ├── reconstruction_loss.pkl
│       ├── final_scores.pkl
│       ├── score_records.jsonl
│       └── scoring_summary.json
└── ...
```

`final_scores.pkl` 是 `sample_id -> score` 字典；`score_records.jsonl` 是更容易和其他模块对齐的逐行 JSON 记录。

### 7. 单模态分数集成

```powershell
python run_single_modality_ensemble.py --split test
```

默认输出：

```text
outputs/vt_tiny_mot_ensemble/
├── aligned_scores_test.csv
├── aligned_scores_test.pkl
├── mean_scores_test.csv
├── mean_scores_test.pkl
├── max_scores_test.csv
├── max_scores_test.pkl
├── mean_score_records_test.jsonl
├── max_score_records_test.jsonl
├── complementary_metrics_test.csv
├── complementary_tau_test.csv
└── ensemble_summary_test.json
```

其中：

- `aligned_scores_*`：按 `sample_id` 对齐后的 6 路 detector 分数。
- `mean_scores_*`：6 路 inverse-rank score 均值集成。
- `max_scores_*`：6 路 inverse-rank score 最大值集成。
- `complementary_*`：baseline-style complementary elimination 过程记录。

### 8. 分析结果

```powershell
python analyze_single_modality_results.py --split test
```

默认输出到：

```text
outputs/vt_tiny_mot_analysis/
```

通常优先看：

- `top_50_test.csv`
- `top_50_sequences_test.csv`
- `top_50_categories_test.csv`
- `analysis_summary_test.json`

### 9. 渲染异常热力图

```powershell
python render_sequence_anomaly_heatmaps.py --split test
```

默认读取：

- `outputs/vt_tiny_mot_ensemble/mean_scores_test.csv`
- `outputs/vt_tiny_mot_individual/individual_trajectories_test.jsonl`
- `../datasets/VT-Tiny-MOT/test2017`

默认输出到：

```text
outputs/vt_tiny_mot_heatmaps/
```

## 快速复用已有结果

如果 `outputs/` 中已经存在完整的模型和中间文件，可以不从头训练。

只重新做 test 集集成：

```powershell
python run_single_modality_ensemble.py --split test
```

只重新分析 test 集结果：

```powershell
python analyze_single_modality_results.py --split test
```

从已有模型重新打分、集成、分析：

```powershell
python score_all_single_modality_detectors.py --split test
python run_single_modality_ensemble.py --split test
python analyze_single_modality_results.py --split test
```

## 群体异常检测预留接口

群体异常检测建议分三步接入：

1. 用 `export_vt_tiny_mot_group_windows.py` 从现有 observations CSV 导出 group/window 输入。
2. 实现一个满足 `GroupAnomalyDetector` Protocol 的 group 模型。
3. 输出 `GroupScoreRecord`，再通过 `aggregate_group_scores_by_sample(...)` 聚合成现有 `ScoreRecord`，交给后续 fusion。

### 导出 group/window 样本

固定窗口模式：

```powershell
python export_vt_tiny_mot_group_windows.py `
  --split train `
  --csv-path outputs/vt_tiny_mot_trajectories/observations_train.csv `
  --sample-mode window `
  --window-size 16 `
  --stride 8
```

完整 sequence 模式：

```powershell
python export_vt_tiny_mot_group_windows.py `
  --split train `
  --csv-path outputs/vt_tiny_mot_trajectories/observations_train.csv `
  --sample-mode sequence
```

默认输出：

```text
outputs/vt_tiny_mot_group/
├── group_windows_train.jsonl
└── group_windows_summary_train.json
```

一个 group window 的核心结构：

```json
{
  "window_id": "DJI_0022_1:0-0",
  "sequence": "DJI_0022_1",
  "frame_start": 0,
  "frame_end": 0,
  "frames": [0],
  "num_objects": 1,
  "objects": [
    {
      "sample_id": "DJI_0022_1:4700000",
      "sequence": "DJI_0022_1",
      "track_id": "4700000",
      "category_id": 0,
      "category_name": "ship",
      "visible_rgb_frames": 1,
      "visible_thermal_frames": 1,
      "states": [
        {
          "frame_id": 0,
          "rgb": {"center_xy": [20.0, 432.5]},
          "thermal": {"center_xy": [22.5, 430.0]},
          "modal": {"offset_distance": 3.5355}
        }
      ]
    }
  ],
  "modalities": ["rgb", "thermal"],
  "feature_names": [
    "cx",
    "cy",
    "x",
    "y",
    "w",
    "h",
    "vx_px_per_frame",
    "vy_px_per_frame",
    "speed_px_per_frame"
  ]
}
```

### Python 接口

后续 group 模型应当对齐这个协议：

```python
from collections.abc import Iterable

from mtf_ba.group_interface import GroupScoreRecord, GroupWindow


class MyGroupDetector:
    def score_windows(
        self,
        windows: Iterable[GroupWindow],
    ) -> Iterable[GroupScoreRecord]:
        for window in windows:
            for obj in window.objects:
                yield GroupScoreRecord(
                    sequence=window.sequence,
                    track_id=obj["track_id"],
                    window_id=window.window_id,
                    frame_start=window.frame_start,
                    frame_end=window.frame_end,
                    category_id=obj["category_id"],
                    category_name=obj["category_name"],
                    score=0.0,
                    component_scores={},
                    metadata={"model": "my_group_detector"},
                )
```

如果模型对同一个 `sample_id` 在多个窗口都输出分数，可以聚合到 object-level：

```python
from mtf_ba.group_interface import aggregate_group_scores_by_sample

object_level_group_scores = aggregate_group_scores_by_sample(
    records=window_level_records,
    method="max",
)
```

聚合后的记录使用现有 `ScoreRecord`，字段约定和个体检测一致：

```json
{
  "sample_id": "DJI_0022_1:4700000",
  "sequence": "DJI_0022_1",
  "track_id": "4700000",
  "source": "group",
  "score": 0.83,
  "component_scores": {
    "group_window_max": 0.83,
    "group_window_mean": 0.52
  },
  "metadata": {
    "aggregation": "max",
    "num_group_windows": 4
  }
}
```

### 后续 fusion 建议

fusion 层建议只依赖 object-level `ScoreRecord`：

- individual 分数：`source = "individual"` 或当前 ensemble 输出。
- group 分数：`source = "group"`。
- 对齐键：`sample_id`。
- 最小可行融合：rank mean / weighted rank mean。
- 后续可扩展融合：学习权重、校准分数、按类别/场景分组融合。

## 关键数据契约

### `sample_id`

所有 object-level 记录统一使用：

```text
sample_id = "{sequence}:{track_id}"
```

构造函数在：

```python
from mtf_ba.schemas import build_sample_id
```

### `ScoreRecord`

标准异常分数记录在 `mtf_ba.schemas.ScoreRecord`：

```json
{
  "sample_id": "DJI_0022_1:4700000",
  "sequence": "DJI_0022_1",
  "track_id": "4700000",
  "category_id": 0,
  "category_name": "ship",
  "source": "individual",
  "score": 0.83,
  "component_scores": {
    "route_rgb": 0.71,
    "speed_rgb": 0.65
  },
  "metadata": {}
}
```

## 常见问题

### 缺少 Python 包

直接安装当前代码导入需要的依赖：

```powershell
pip install numpy pandas torch scikit-learn scipy tqdm kneed matplotlib
```

### GPU 不可用

检查：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，可以先用 CPU 跑通流程；训练会慢，但逻辑一致。

### `No training sequences remain after filtering`

说明某一路特征在长度过滤后没有样本。常见于 `shape_*`。可以降低最小长度：

```powershell
python train_all_single_modality_detectors.py --min-length 2
```

### 找不到 test 文件

确认已经按顺序生成：

```powershell
python extract_vt_tiny_mot_trajectories.py --split test
python export_vt_tiny_mot_individual_trajectories.py --split test --csv-path outputs/vt_tiny_mot_trajectories/observations_test.csv
python export_single_modality_features.py --jsonl-path outputs/vt_tiny_mot_individual/individual_trajectories_test.jsonl --split test
```

## 推荐开发顺序

当前阶段继续开发时，建议保持这个顺序：

1. 先稳定个体异常检测结果和评价方式。
2. 用 `export_vt_tiny_mot_group_windows.py` 固定 group 输入数据。
3. 实现第一个简单 group baseline，例如窗口内对象速度/相对位置统计异常。
4. 输出 `GroupScoreRecord`，聚合到 object-level group score。
5. 再实现 individual + group fusion。

这样可以保证后续群体异常模块接入时，不需要改动已经完成的个体轨迹、特征和打分链路。
