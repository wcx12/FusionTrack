# FusionTrack

FusionTrack 是一个面向多模态目标融合、轨迹补全与异常检测的毕业设计/论文项目。仓库包含论文 LaTeX 源码、系统可视化材料，以及基于 VT-Tiny-MOT 数据集构建的个体级和群体级异常检测基准实验。

本 README 是项目总入口，说明仓库结构、实验协议、方法分类、当前结果、复现实验方式，以及哪些文件适合提交到 GitHub。

## 当前状态

- 论文源码位于 `article_content/`，使用 XeLaTeX/latexmk 构建。
- 异常检测基准实验位于 `code/anomaly_detection/benchmark/`，覆盖个体级与群体级任务。
- 当前 strict validation protocol 使用 seed 42、train split 内部 train/val 划分。
- 个体级主键是 `sample_id`，严格要求 829 条 label 与 829 条 score 完整对齐。
- 群体级主键是 `sample_id + window_id`，严格要求 15605 条 label 与 15605 条 score 完整对齐。
- 最新本地矩阵增强结果在远程 `remote_runs/fusiontrack_b3b8599_methods_20260522`，本地归档在 `server_artifacts/final_results_20260522/fusiontrack_update_results_20260522.tar.gz`。
- 最新官方 long-budget baseline 结果在远程 `remote_runs/fusiontrack_b3b8599_convergence_20260522`。
- 最新 recent official baseline 结果在远程 `remote_runs/fusiontrack_recent_official_20260522`，本地归档在 `server_artifacts/final_results_20260522/fusiontrack_recent_official_20260522.tar.gz`。
- CETrajAD full-coverage strict rerun 在远程 `remote_runs/fusiontrack_cetrajad_fullcoverage_20260522`，本地归档在 `server_artifacts/final_results_20260522/fusiontrack_cetrajad_fullcoverage_20260522.tar.gz`。

## 仓库结构

```text
FusionTrack/
├── article_content/                         # 论文 LaTeX 源码、图片、字体、参考文献
├── code/
│   ├── anomaly_detection/
│   │   ├── benchmark/                       # 当前异常检测实验主入口
│   │   │   ├── baselines/                   # 经典 baseline 与 internal proxy
│   │   │   ├── configs/                     # 实验设置、审计记录、最终协议说明
│   │   │   ├── evaluation/                  # 指标、结果读取、表格导出
│   │   │   ├── external_sources/            # 官方论文源码 adapter
│   │   │   ├── fusiontrack/                 # FusionTrack 个体/群体方法
│   │   │   ├── policies/                    # 论文源码复现规则
│   │   │   ├── protocol/                    # split、异常注入、schema
│   │   │   ├── runners/                     # 数据准备、训练、评价、审计脚本
│   │   │   └── tests/                       # benchmark 单元测试
│   │   └── individual/                      # 早期个体级异常检测代码
│   ├── registration/                        # MPS-GAF 配准相关代码
│   └── vpr/                                 # TF-VPR 相关代码
├── server_artifacts/                        # 服务器实验归档，不提交
├── visualization_results/                   # 系统可视化结果与演示材料
└── .github/workflows/build-article.yml      # 论文 PDF 构建 workflow
```

## 实验协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | VT-Tiny-MOT |
| source split | `train` |
| validation ratio | `0.2` |
| seed | `42` |
| individual anomaly fraction | `0.1` |
| group anomaly fraction | `0.1` |
| group window / stride | `16 / 8` |
| individual key | `sample_id` |
| group key | `sample_id + window_id` |
| rank metric | `P@100 / R@100` |

公平性规则：

- 所有方法使用同一份 generated protocol。
- test split 不用于调参。
- 论文型 baseline 必须使用官方源码或论文明确指向的源码复现，并记录 URL、commit、license、adapter、环境和 run manifest。
- 本地 proxy 不能直接写成 CETrajAD、LM-TAD、Pi-DPM 等原论文方法名。
- deep baseline 不强制相同 epoch，而是记录各自预算、batch size、learning rate、window size、seed、GPU 和收敛状态。
- 进入主结果的 score 文件必须通过 strict key audit：无重复 label、无重复 score、无缺失 score、无多余 score。

## 方法分类

### 我们的方法

| 方法 | 任务 | 类型 | 说明 |
| --- | --- | --- | --- |
| `fusiontrack_individual_nn` | 个体级 | 学习型 | 手工轨迹特征 nearest-neighbor profile |
| `fusiontrack_individual_ensemble` | 个体级 | 学习型 | nearest-feature、LOF novelty、Isolation Forest 的无标签 rank ensemble |
| `fusiontrack_individual_context` | 个体级 | 学习型 | 加入群体上下文特征的 nearest-neighbor profile |
| `fusiontrack_group_graph` | 群体级 | 非学习型 | 图关系、相对运动和群体事件规则打分 |
| `fusiontrack_group_temporal_knn` | 群体级 | 学习型 | 群体窗口特征 KNN |
| `fusiontrack_group_hybrid` | 群体级 | 学习型/融合型 | prediction residual、graph cohesion、temporal profile 的 rank fusion |

### 经典基线方法

| 方法 | 任务 | 类型 |
| --- | --- | --- |
| `individual_lof` | 个体级 | 学习型，classical ML |
| `individual_iforest` | 个体级 | 学习型，classical ML |
| `individual_ocsvm` | 个体级 | 学习型，classical ML |
| `group_prediction_linear` | 群体级 | 非学习型，linear prediction residual |
| `group_lof` | 群体级 | 学习型，classical ML |
| `group_iforest` | 群体级 | 学习型，classical ML |
| `group_ocsvm` | 群体级 | 学习型，classical ML |

### 官方论文基线方法

这些方法必须使用论文官方或论文明确关联的 GitHub 源码复现。

| 方法 | 当前表中名称 | 任务 | 状态 |
| --- | --- | --- | --- |
| LM-TAD | `official_lmtad` / `official_lmtad_50` | 个体级 | 已接入官方源码；50 epoch long-budget 已收敛 |
| Pi-DPM | `official_pidpm` | 个体级 | 已接入官方源码；原 run 已收敛 |
| TranAD | `official_tranad` / `official_tranad_50` | 个体级、群体级 | 已接入官方源码；50 epoch 仍未收敛 |
| Anomaly Transformer | `official_anomaly_transformer` / `official_anomaly_transformer_50` | 个体级、群体级 | 已接入官方源码；50 epoch 仍未收敛 |
| DCdetector | `official_dcdetector` | 个体级、群体级 | 已接入官方源码；原 run 已收敛 |
| CETrajAD | `official_cetrajad_fullcoverage` | 个体级 | 原始官方流程 coverage failed；full-coverage adapter rerun 已通过 829/829 strict key audit |
| CATCH | `official_catch_*_20` | 个体级、群体级 | 已接入官方源码；20 epoch recent run 已完成 |
| CutAddPaste | `official_cutaddpaste_*_20` | 个体级、群体级 | 已接入官方源码；20 epoch recent run 已完成 |
| TimeMixer | `official_timemixer_*_20` | 个体级、群体级 | 已接入官方源码；20 epoch recent run 已完成 |
| SensitiveHUE | `official_sensitive_hue_*_20` | 个体级、群体级 | 补充 official-source 候选；公开 README 仍标为 under review |

### 内部方法、代理方法与消融

| 方法 | 任务 | 说明 |
| --- | --- | --- |
| `individual_complementary_cetrajad_proxy` | 个体级 | CETrajAD 思路的本地 proxy，不能写成官方 CETrajAD |
| `individual_physics_kinematic_proxy` | 个体级 | 物理/运动学先验 proxy |
| `individual_trajectory_lm_ngram_proxy` | 个体级 | trajectory language model 的 n-gram proxy |
| `group_temporal_graph_ae_proxy` | 群体级 | PCA reconstruction proxy |

## 当前验证集结果

结果来自 strict validation protocol。指标包括 AUROC、AUPRC、F1、P@100 和 R@100。

### 个体级

| Method | Category | Learning | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_individual_ensemble` | Our method | Learning | 0.623752 | 0.153147 | 0.254642 | 0.150000 | 0.180723 |
| `individual_lof` | Classic baseline | Learning | 0.606512 | 0.159437 | 0.246014 | 0.160000 | 0.192771 |
| `fusiontrack_individual_nn` | Our method | Learning | 0.595626 | 0.134131 | 0.237681 | 0.180000 | 0.216867 |
| `individual_iforest` | Classic baseline | Learning | 0.592445 | 0.127709 | 0.218310 | 0.160000 | 0.192771 |
| `individual_physics_kinematic_proxy` | Internal proxy | Learning | 0.578410 | 0.126905 | 0.213740 | 0.170000 | 0.204819 |
| `individual_complementary_cetrajad_proxy` | Internal proxy | Learning | 0.567137 | 0.122853 | 0.224793 | 0.130000 | 0.156627 |
| `fusiontrack_individual_context` | Our method | Learning | 0.561275 | 0.120676 | 0.204348 | 0.130000 | 0.156627 |
| `official_catch_individual_20` | Official paper baseline | Learning | 0.543759 | 0.124095 | 0.204793 | 0.120000 | 0.144578 |
| `official_anomaly_transformer_50` | Official paper baseline | Learning | 0.530217 | 0.110745 | 0.191740 | 0.120000 | 0.144578 |
| `official_sensitive_hue_individual_20` | Supplementary official-source | Learning | 0.524387 | 0.110813 | 0.196796 | 0.100000 | 0.120482 |
| `official_timemixer_individual_20` | Official paper baseline | Learning | 0.521658 | 0.113224 | 0.197441 | 0.090000 | 0.108434 |
| `official_cetrajad_fullcoverage` | Official paper baseline | Learning | 0.521092 | 0.106465 | 0.193437 | 0.080000 | 0.096386 |
| `official_lmtad_50` | Official paper baseline | Learning | 0.474708 | 0.100031 | 0.183374 | 0.070000 | 0.084337 |
| `official_cutaddpaste_individual_20` | Official paper baseline | Learning | 0.472916 | 0.128043 | 0.182222 | 0.090000 | 0.108434 |
| `official_tranad_50` | Official paper baseline | Learning | 0.455926 | 0.103066 | 0.187283 | 0.130000 | 0.156627 |

### 群体级

| Method | Category | Learning | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_group_hybrid` | Our method | Learning/Fusion | 0.692177 | 0.081551 | 0.193548 | 0.150000 | 0.170455 |
| `official_timemixer_group_20` | Official paper baseline | Learning | 0.627657 | 0.014647 | 0.033520 | 0.030000 | 0.034091 |
| `group_prediction_linear` | Classic baseline | Non-learning | 0.622238 | 0.093898 | 0.162319 | 0.100000 | 0.113636 |
| `official_catch_group_20` | Official paper baseline | Learning | 0.605357 | 0.019403 | 0.022472 | 0.010000 | 0.011364 |
| `official_anomaly_transformer_50` | Official paper baseline | Learning | 0.575085 | 0.013800 | 0.029268 | 0.010000 | 0.011364 |
| `official_cutaddpaste_group_20` | Official paper baseline | Learning | 0.561328 | 0.020299 | 0.058252 | 0.040000 | 0.045455 |
| `official_sensitive_hue_group_20` | Supplementary official-source | Learning | 0.467138 | 0.020232 | 0.046823 | 0.030000 | 0.034091 |
| `group_lof` | Classic baseline | Learning | 0.417786 | 0.005899 | 0.031332 | 0.020000 | 0.022727 |
| `official_tranad_50` | Official paper baseline | Learning | 0.410594 | 0.004745 | 0.013699 | 0.000000 | 0.000000 |
| `fusiontrack_group_graph` | Our method | Non-learning | 0.361618 | 0.006877 | 0.041152 | 0.030000 | 0.034091 |
| `fusiontrack_group_temporal_knn` | Our method | Learning | 0.302068 | 0.005664 | 0.029412 | 0.020000 | 0.022727 |

`fusiontrack_group_hybrid` 当前在 AUROC、F1、P@100 和 R@100 上最好；`group_prediction_linear` 当前在 AUPRC 上最好。最终论文结论需要在 test split 或多 seed 上确认。

## 复现实验

运行 benchmark 单元测试：

```bash
python -c "import collections, collections.abc, pytest; collections.Callable = collections.abc.Callable; raise SystemExit(pytest.main(['code/anomaly_detection/benchmark/tests']))"
```

生成 protocol 数据：

```bash
python code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_protocol.py \
  --data-root /path/to/VT-Tiny-MOT \
  --output-root code/anomaly_detection/benchmark/outputs/protocol \
  --source-split train \
  --seed 42
```

运行 benchmark matrix：

```bash
python code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --config-json code/anomaly_detection/benchmark/outputs/protocol/individual_val_matrix.json \
  --output-dir code/anomaly_detection/benchmark/outputs/results/individual

python code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --config-json code/anomaly_detection/benchmark/outputs/protocol/group_val_matrix.json \
  --output-dir code/anomaly_detection/benchmark/outputs/results/group
```

单独评价 score 文件：

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

## 服务器与 GPU 实验

长时间训练建议使用 `tmux`，避免 SSH 断开导致实验中断：

```bash
USE_TMUX=1 \
TMUX_SESSION=fusiontrack_val \
MODE=val \
GPU_ID=0 \
SEED=42 \
DATA_ROOT=data/VT-Tiny-MOT \
OUTPUT_ROOT=remote_runs/fusiontrack_val_strict/protocol \
RESULT_ROOT=remote_runs/fusiontrack_val_strict/results \
bash code/anomaly_detection/benchmark/runners/run_server_gpu_experiments.sh
```

不要把服务器密码、私钥、token、原始数据或模型权重写入 README、配置、日志或提交记录。

## 论文构建

```bash
cd article_content
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

生成的 PDF 和 LaTeX 中间产物不应提交。清理命令：

```bash
cd article_content
latexmk -C
```

## 可提交与不可提交内容

建议提交：

- `code/anomaly_detection/benchmark/` 中的 runner、adapter、evaluation、protocol、method 实现和测试。
- `code/anomaly_detection/benchmark/configs/` 与 `code/anomaly_detection/benchmark/policies/` 中的小型配置和规则文档。
- `code/registration/` 中的 MPS-GAF、非学习基线、外部学习式基线 adapter、schema 转换脚本、协议记录和测试。
- `tests/registration/` 中面向配准转换、数据导出和路径策略的测试。
- 论文 LaTeX 源码、必要图片、参考文献和 workflow。
- README、实验说明、可复现脚本和小型表格。

不建议提交：

- `server_artifacts/`、`output/`、`runs/`、`remote_runs/`、`logs/`、`checkpoints/`。
- 原始数据集、远程服务器中间产物、大模型权重、`.pt/.pth/.pkl/.npy/.npz`。
- `external_src/`、`transfer_*.zip`、第三方源码压缩包和本地传输包。
- `article_content/main.pdf` 与 LaTeX 中间产物；正式 PDF 建议由本地或 GitHub Actions 构建生成。
- 密码、私钥、token 或带敏感信息的服务器登录命令。
- 第三方官方 baseline 仓库源码；优先记录 URL、commit、license，并通过 adapter 外部调用。

## 已知限制与下一步

- `fusiontrack_individual_ensemble` 当前 AUROC 和 F1 最好，但 AUPRC 与 P@100/R@100 仍不是最优。
- `fusiontrack_group_hybrid` 当前总体最强，但 AUPRC 仍低于 `group_prediction_linear`。
- CETrajAD 原始官方流程存在 770/829 coverage 问题；已补跑 `official_cetrajad_fullcoverage`，可以进入 strict 对比表，但必须注明 full-coverage adapter、`coordinate_scale=1.0` 和 `no-loss-history` 收敛状态。
- TranAD 与 Anomaly Transformer 的 50 epoch long-budget run 仍未收敛；若论文需要完全收敛结论，需要继续扩展预算。
- 当前结果来自 seed 42 validation protocol，最终论文建议补 test split 或多 seed。
