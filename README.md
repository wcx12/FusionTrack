# FusionTrack

FusionTrack 是一个面向多模态目标融合、轨迹补全与异常检测的毕业设计/论文项目。当前仓库包含论文 LaTeX 源码、系统可视化材料，以及基于 VT-Tiny-MOT 数据集构建的个体级与群体级异常检测 benchmark。

本 README 是项目总入口：说明仓库结构、实验协议、方法分类、当前结果、复现实验方式，以及哪些文件适合提交到 GitHub。

## 当前状态

- 论文源码位于 `article_content/`，通过 XeLaTeX/latexmk 构建。
- 异常检测 benchmark 位于 `code/anomaly_detection/benchmark/`，已经覆盖个体级与群体级任务。
- 当前已完成严格对齐的验证集实验：个体级每个方法 829 条 score，群体级每个方法 15605 条 score。
- 最新本地矩阵远程重跑为服务器 `tmux` 会话 `fusiontrack_group_knn_b3b8599`，源码 revision 为 `b3b8599`，结果目录为 `/root/autodl-tmp/fusiontrack_b3b8599_val/results`。
- 最新官方论文 baseline 远程 GPU 重跑结果位于 `/root/autodl-tmp/fusiontrack_b3b8599_official_20260522`，其中 strict 主表结果已通过 0 重复、0 缺失、0 多余 score 审计。
- 历史官方论文 baseline 汇总结果保存在本地 `server_artifacts/final_results_20260521/`。该目录用于保存服务器/本地实验产物，默认不提交到 GitHub。
- 官方论文 baseline 的原则是：论文主表中只把来自论文官方或论文明确关联源码的结果写成原论文方法名；本地近似实现只能作为 internal/proxy/ablation。

## 仓库结构

```text
FusionTrack/
├── article_content/                         # 论文 LaTeX 源码、图片、字体、参考文献
├── code/
│   ├── anomaly_detection/
│   │   ├── benchmark/                       # 当前异常检测实验主入口
│   │   │   ├── baselines/                   # 经典 baseline 与 proxy/ablation
│   │   │   ├── configs/                     # 实验配置、最终协议设置、官方源码记录模板
│   │   │   ├── evaluation/                  # 指标、结果读取、表格导出
│   │   │   ├── external_sources/            # 官方论文源码 adapter
│   │   │   ├── fusiontrack/                 # FusionTrack 个体/群体方法
│   │   │   ├── policies/                    # 论文源码复现与提交范围规则
│   │   │   ├── protocol/                    # split、异常注入、schema
│   │   │   ├── runners/                     # 数据准备、训练、评价、审计脚本
│   │   │   └── tests/                       # benchmark 单元测试
│   │   └── individual/                      # 早期个体级异常检测代码
│   ├── registration/                        # MPS-GAF 配准相关代码
│   └── vpr/                                 # TF-VPR 相关代码
├── docs/                                    # 本地计划、阶段报告与实验说明
├── server_artifacts/                        # 本地/服务器实验结果，不提交
├── visualization_results/                   # 系统可视化结果、图像和演示材料
└── .github/workflows/build-article.yml      # 论文 PDF 构建 workflow
```

## 异常检测 Benchmark

当前 benchmark 使用 VT-Tiny-MOT 数据，按统一协议构造个体级和群体级异常检测任务。

### 任务定义

| 任务 | 评价对象 | 主键 | 当前严格对齐规模 |
| --- | --- | --- | ---: |
| 个体级异常检测 | 单个目标轨迹/样本 | `sample_id` | 829 |
| 群体级异常检测 | 群体时间窗口 | `sample_id + window_id` | 15605 |

群体级主协议使用 `sample_id + window_id` 作为唯一键。旧的 `sample_id` any-window 聚合只适合放在附录或诊断分析中，不应替代主表结果。

### 公平性规则

- 所有方法使用同一份 train/validation protocol 文件。
- 测试集不用于调参。
- 论文 baseline 必须记录官方源码 URL、commit、license、adapter、环境、split 和 run manifest。
- 本地 proxy 不能直接写成 CETrajAD、LM-TAD、Pi-DPM 等原论文方法名。
- 深度方法不强制与非深度方法使用相同 epoch；应使用各自合理训练预算，并记录 epoch、batch size、learning rate、window size、seed、GPU、收敛状态。
- 进入主表的 score 文件必须通过严格审计：无重复 label、无重复 score、无缺失 score、无多余 score。

## 方法分类

### 我们的方法

| 方法名 | 任务 | 类型 | 说明 |
| --- | --- | --- | --- |
| `fusiontrack_individual_nn` | 个体级 | 学习型，nearest-neighbor profile | 当前主要个体级 FusionTrack 方法 |
| `fusiontrack_individual_context` | 个体级 | 学习型，context-aware nearest-neighbor profile | 加入群体上下文的个体级方法 |
| `fusiontrack_group_temporal_knn` | 群体级 | 学习型，standardized group-feature KNN | 学习型群体级 FusionTrack 方法，已在远程严格验证协议下重跑 |
| `fusiontrack_group_graph` | 群体级 | 非学习型，graph/rule scoring | 当前群体级 FusionTrack 方法 |

说明：`fusiontrack_group_graph` 是非学习型图关系/规则打分方法；`fusiontrack_group_temporal_knn` 是后续补充的学习型群体级方法，已经在服务器严格 full-coverage 协议下生成验证集结果。

### 官方论文 Baseline

这些方法应尽量使用论文官方或论文明确关联的 GitHub 源码复现，并通过 adapter 接入统一评价协议。

| 方法名 | 当前表中名称 | 任务 | 状态 |
| --- | --- | --- | --- |
| LM-TAD | `official_lmtad` | 个体级 | 已接入官方源码适配结果 |
| Pi-DPM | `official_pidpm` | 个体级 | 已接入官方源码适配结果 |
| TranAD | `official_tranad` | 个体级/群体级 | 已接入官方源码适配结果 |
| Anomaly Transformer | `official_anomaly_transformer` | 个体级/群体级 | 已接入官方源码适配结果 |
| DCdetector | `official_dcdetector` | 个体级/群体级 | 已接入官方源码适配结果 |
| CETrajAD | 暂不进主表 | 个体级 | 官方流程覆盖失败，保留为 coverage-failed 记录 |

CETrajAD 当前不放入严格 full-coverage 主表：稳定 adapter 只能输出 770/829 条分数，缺失 59 条零运动/短轨迹；放宽 adapter 后官方 PCA 预处理出现 NaN。为了不篡改数据，目前只保留 coverage-failed 说明。

### 经典 Baseline

| 方法名 | 任务 | 类型 |
| --- | --- | --- |
| `individual_lof` | 个体级 | 学习型，classical ML |
| `individual_iforest` | 个体级 | 学习型，classical ML |
| `individual_ocsvm` | 个体级 | 学习型，classical ML |
| `group_prediction_linear` | 群体级 | 非学习型，linear prediction residual |
| `group_lof` | 群体级 | 学习型，classical ML |
| `group_iforest` | 群体级 | 学习型，classical ML |
| `group_ocsvm` | 群体级 | 学习型，classical ML |

### Internal / Proxy / Ablation

| 方法名 | 任务 | 说明 |
| --- | --- | --- |
| `individual_complementary_cetrajad_proxy` | 个体级 | CETrajAD 思路的本地 proxy，不能写成官方 CETrajAD |
| `individual_physics_kinematic_proxy` | 个体级 | 物理/运动学先验 proxy |
| `individual_trajectory_lm_ngram_proxy` | 个体级 | trajectory language model 的 n-gram proxy |
| `group_temporal_graph_ae_proxy` | 群体级 | PCA reconstruction proxy |

## 当前验证集结果

本地矩阵结果来自远程服务器重跑 `/root/autodl-tmp/fusiontrack_b3b8599_val/results`，源码 revision 为 `b3b8599`。官方论文 baseline 来自 2026-05-22 远程 GPU 重跑 `/root/autodl-tmp/fusiontrack_b3b8599_official_20260522`。GPU 环境为 NVIDIA GeForce RTX 5090。指标包括 AUROC、AUPRC、F1、P@100 和 R@100。CETrajAD 官方流程只输出 770/829 条 score，保留为 coverage-failed 诊断，不进入 strict 主表。

### 个体级结果

| Method | Category | Learning | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `individual_lof` | Classic baseline | Learning | 0.606512 | 0.159437 | 0.246014 | 0.160000 | 0.192771 |
| `fusiontrack_individual_nn` | Our method | Learning | 0.595626 | 0.134131 | 0.237681 | 0.180000 | 0.216867 |
| `individual_iforest` | Classic baseline | Learning | 0.592445 | 0.127709 | 0.218310 | 0.160000 | 0.192771 |
| `individual_physics_kinematic_proxy` | Internal proxy/ablation | Learning | 0.578410 | 0.126905 | 0.213740 | 0.170000 | 0.204819 |
| `individual_complementary_cetrajad_proxy` | Internal proxy/ablation | Learning | 0.567137 | 0.122853 | 0.224793 | 0.130000 | 0.156627 |
| `fusiontrack_individual_context` | Our method | Learning | 0.561275 | 0.120676 | 0.204348 | 0.130000 | 0.156627 |
| `official_dcdetector` | Official paper baseline | Learning | 0.523095 | 0.104621 | 0.190210 | 0.430000 | 0.518072 |
| `official_anomaly_transformer` | Official paper baseline | Learning | 0.493427 | 0.106067 | 0.183223 | 0.130000 | 0.156627 |
| `individual_ocsvm` | Classic baseline | Learning | 0.484043 | 0.098442 | 0.185780 | 0.080000 | 0.096386 |
| `official_pidpm` | Official paper baseline | Learning | 0.459236 | 0.092592 | 0.182618 | 0.110000 | 0.132530 |
| `official_tranad` | Official paper baseline | Learning | 0.443619 | 0.097967 | 0.186308 | 0.080000 | 0.096386 |
| `official_lmtad` | Official paper baseline | Learning | 0.434462 | 0.086565 | 0.184721 | 0.070000 | 0.084337 |
| `individual_trajectory_lm_ngram_proxy` | Internal proxy/ablation | Learning | 0.419014 | 0.082957 | 0.185520 | 0.060000 | 0.072289 |

### 群体级结果

| Method | Category | Learning | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `group_prediction_linear` | Classic baseline | Non-learning | 0.622238 | 0.093898 | 0.162319 | 0.100000 | 0.113636 |
| `official_anomaly_transformer` | Official paper baseline | Learning | 0.601933 | 0.008517 | 0.021175 | 0.010000 | 0.011364 |
| `official_dcdetector` | Official paper baseline | Learning | 0.504306 | 0.005744 | 0.013201 | 0.060000 | 0.068182 |
| `official_tranad` | Official paper baseline | Learning | 0.423703 | 0.004994 | 0.017391 | 0.000000 | 0.000000 |
| `group_lof` | Classic baseline | Learning | 0.417786 | 0.005899 | 0.031332 | 0.020000 | 0.022727 |
| `group_temporal_graph_ae_proxy` | Internal proxy/ablation | Learning | 0.371188 | 0.005975 | 0.021277 | 0.010000 | 0.011364 |
| `fusiontrack_group_graph` | Our method | Non-learning | 0.361618 | 0.006877 | 0.041152 | 0.030000 | 0.034091 |
| `fusiontrack_group_temporal_knn` | Our method | Learning | 0.302068 | 0.005664 | 0.029412 | 0.020000 | 0.022727 |
| `group_ocsvm` | Classic baseline | Learning | 0.261264 | 0.004887 | 0.030675 | 0.020000 | 0.022727 |
| `group_iforest` | Classic baseline | Learning | 0.207463 | 0.003280 | 0.011258 | 0.000000 | 0.000000 |

## 复现实验

### 1. 运行测试

当前 benchmark 测试在本地通过。若遇到旧版 `collections.Callable` 兼容问题，可使用下面的入口运行：

```bash
python -c "import collections, collections.abc, pytest; collections.Callable = collections.abc.Callable; raise SystemExit(pytest.main(['code/anomaly_detection/benchmark/tests']))"
```

最近一次完整 benchmark 测试结果为 `132 passed, 1 warning`。warning 来自 pandas/bottleneck 版本提示，不是代码失败。

### 2. 生成协议数据

```bash
python code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_protocol.py \
  --data-root /path/to/VT-Tiny-MOT \
  --output-root code/anomaly_detection/benchmark/outputs/protocol \
  --split val \
  --seed 42
```

具体参数以 `code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_protocol.py --help` 为准。

### 3. 运行本地 benchmark matrix

```bash
python code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --config-json code/anomaly_detection/benchmark/configs/vt_tiny_mot_matrix.example.json \
  --output-dir code/anomaly_detection/benchmark/outputs/local_seed42
```

注意：`vt_tiny_mot_matrix.example.json` 是示例配置。最终论文实验应优先使用 `code/anomaly_detection/benchmark/configs/final_experiment_settings.json` 对应的协议和参数。

### 4. 单独评价 score 文件

```bash
python code/anomaly_detection/benchmark/runners/run_evaluation.py \
  --score-path path/to/scores.jsonl \
  --label-path path/to/labels.jsonl \
  --key-fields sample_id window_id \
  --k 100 \
  --require-unique-keys \
  --require-score-key-match \
  --output-json path/to/metrics.json
```

个体级通常使用 `--key-fields sample_id`；群体级主协议使用 `--key-fields sample_id window_id`。

### 5. 审计并合并结果

```bash
python code/anomaly_detection/benchmark/runners/audit_and_merge_results.py \
  --label-path path/to/labels.jsonl \
  --score-dir path/to/scores \
  --key-fields sample_id window_id \
  --k 100 \
  --require-unique-keys \
  --require-score-key-match \
  --output-csv path/to/summary.csv \
  --output-json path/to/summary.json
```

## 服务器与 GPU 实验

长时间训练建议在服务器使用 `tmux`，避免 SSH 断开导致实验中止：

```bash
USE_TMUX=1 \
TMUX_SESSION=fusiontrack_val \
MODE=val \
GPU_ID=0 \
SEED=42 \
DATA_ROOT=/root/FusionTrack/data/VT-Tiny-MOT \
OUTPUT_ROOT=/root/autodl-tmp/fusiontrack_val_strict/protocol \
RESULT_ROOT=/root/autodl-tmp/fusiontrack_val_strict/results \
bash code/anomaly_detection/benchmark/runners/run_server_gpu_experiments.sh
```

断线后重新连接：

```bash
tmux attach -t fusiontrack_val
```

不要把服务器密码、私钥、token、数据集路径中的敏感信息写进 README、配置文件、日志或提交记录。

## 论文构建

论文使用 XeLaTeX 和 `latexmk` 构建：

```bash
cd article_content
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

生成的 PDF 为 `article_content/main.pdf`。构建产物如 `main.pdf`、`.aux`、`.log`、`.xdv`、SyncTeX 文件不应提交。

清理构建产物：

```bash
cd article_content
latexmk -C
```

GitHub Actions 会在论文相关文件变化时构建 PDF，并把结果上传为 workflow artifact，而不是直接存入仓库。

## 可提交与不可提交内容

建议提交：

- 源码：`code/anomaly_detection/benchmark/` 中的 runner、adapter、evaluation、protocol、method 实现和测试。
- 配置模板与小型协议说明：`code/anomaly_detection/benchmark/configs/`、`policies/`。
- 论文 LaTeX 源码、必要图片、参考文献和 workflow。
- README、实验说明、可复现脚本和小型表格。

不建议提交：

- `server_artifacts/`、`output/`、`runs/`、`logs/`、`checkpoints/`。
- 原始数据集、远程服务器中间产物、大模型权重、`.pt/.pth/.pkl/.npy/.npz`。
- 密码、私钥、token、服务器登录命令中的敏感信息。
- 第三方官方 baseline 仓库源码，除非 license 明确允许并且确实需要 vendoring；更推荐记录 URL、commit、license，并通过 adapter 外部调用。

## 已知限制与下一步

- 当前个体级 FusionTrack 方法接近但未超过 `individual_lof` 的 AUROC；需要继续做特征、融合权重和异常注入类型的消融。
- 当前已补充学习型 `fusiontrack_group_temporal_knn`，但它还需要在严格 full-coverage 协议下和 `group_prediction_linear`、官方 TSAD baseline 公平重跑后才能进入主结果表。
- CETrajAD 官方流程目前存在 full-coverage 问题，不进入主表，除非后续能在不改动数据语义的前提下解决官方预处理失败。
- 当前结果是 seed 42 validation protocol；最终论文表建议补充多 seed 或 test split 锁定评估。
