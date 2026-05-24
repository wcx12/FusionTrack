# FusionTrack VT-Tiny-MOT 异常检测实验说明

这份 README 是当前项目的“实验版本完整说明”。它覆盖我已经整理、实现、运行和验证过的全部实验内容，包括实验协议、数据与异常注入、个体级和群体级方法、基线方法分类、官方论文源码复现规则、运行命令、当前结果、可提交内容和后续工作边界。

本文档对应目录：

```text
code/anomaly_detection/benchmark/
```

如果后续要写论文、补实验、提交 GitHub 或继续在服务器上跑结果，应优先以这份 README 和 `configs/final_experiment_settings.md` 为准。

## 1. 当前阶段结论

当前项目已经完成了从“本地验证集调试”到“train -> test holdout 多种子确认”的一版完整实验闭环。

主要结论如下：

1. 个体级异常检测已经有完整实现、经典 baseline、FusionTrack ensemble、校准版本和 validation-tuned 版本。
2. 群体级异常检测已经补齐，包含 prediction residual、graph cohesion、temporal profile、hybrid fusion、residual gate 和多组 validation-tuned 权重。
3. 评价协议已经从早期宽松统计改为严格 key 对齐：
   - 个体级使用 `sample_id`。
   - 群体级主协议使用 `sample_id + window_id`。
   - 每个 label key 必须恰好有一个 score key，重复、缺失和多余 score 都要失败或显式标注。
4. 当前 holdout 多种子实验已经在 VT-Tiny-MOT `train -> test` 协议下完成，seeds 为 `42, 43, 44`。
5. 群体级 FusionTrack 方法明显强于当前经典群体 baseline。
6. 个体级 FusionTrack tuned 版本在 AUROC、F1、P@100、R@100 上最好，但 AUPRC 当前仍略低于 `individual_lof`，差距约 `0.001061`。这意味着后续不能在 test 上继续调参，必须回到 validation 选择新配置，再做 fresh holdout 确认。
7. 论文主表中的“论文型 baseline”必须来自官方或论文链接源码。本地 proxy 只能进入 internal/proxy/ablation 表，不能直接命名为 CETrajAD、LM-TAD 或 Pi-DPM。

## 2. 目录结构和职责

核心目录如下：

```text
code/anomaly_detection/benchmark/
  configs/
    final_experiment_settings.json
    final_experiment_settings.md
  fusiontrack/
    individual_scoring.py
    group_temporal_profile.py
  runners/
    prepare_vt_tiny_mot_protocol.py
    prepare_vt_tiny_mot_holdout_protocol.py
    run_benchmark_matrix.py
    run_fusiontrack_score_grid.py
    run_fusiontrack_holdout_multiseed.py
    run_server_gpu_experiments.sh
  tests/
    test_individual_fusiontrack_scoring.py
    test_group_temporal_profile.py
    test_run_benchmark_matrix.py
    test_prepare_vt_tiny_mot_protocol.py
    test_holdout_multiseed_runner.py
```

各文件作用：

| 路径 | 作用 |
| --- | --- |
| `configs/final_experiment_settings.json` | 机器可读的最终实验配置，记录统一协议、方法列表、预算、官方 baseline、已完成和未完成 rerun。 |
| `configs/final_experiment_settings.md` | 人类可读的最终实验配置，论文和后续实验优先看这个文件。 |
| `fusiontrack/individual_scoring.py` | 个体级 FusionTrack scoring 实现，包括 nearest feature、LOF novelty、Isolation Forest rank ensemble，以及新增的 feature-stratified rank calibration。 |
| `fusiontrack/group_temporal_profile.py` | 群体级 FusionTrack scoring 实现，包括 prediction residual、graph cohesion、temporal profile、hybrid fusion，以及新增 residual gate。 |
| `runners/prepare_vt_tiny_mot_protocol.py` | 生成 validation 协议、异常标签、score 路径和 benchmark matrix。 |
| `runners/prepare_vt_tiny_mot_holdout_protocol.py` | 生成 train -> test holdout 协议；训练只来自 train，异常注入和评价来自 test。 |
| `runners/run_benchmark_matrix.py` | 统一运行 individual/group benchmark matrix，负责把配置参数传给对应 scorer，并输出 summary。 |
| `runners/run_fusiontrack_score_grid.py` | 使用已缓存组件分数做快速权重搜索，不重复计算底层组件。 |
| `runners/run_fusiontrack_holdout_multiseed.py` | 多 seed holdout 总控脚本，循环 prepare、运行 matrix、聚合 `all_runs.csv`、`aggregate.csv` 和 `best_by_metric.json`。 |
| `runners/run_server_gpu_experiments.sh` | 服务器端 tmux/GPU 实验入口脚本。 |
| `tests/` | 覆盖新增 calibration、residual gate、runner 参数传递、protocol 生成、holdout 聚合等行为。 |

服务器结果归档目录：

```text
server_artifacts/final_results_20260522/
```

当前最重要的本地归档结果：

```text
server_artifacts/final_results_20260522/holdout_multiseed_20260522/
  fusiontrack_holdout_multiseed_combined_20260522/
    aggregate.csv
    all_runs.csv
    best_by_metric.json
    manifest.json
```

## 3. 数据集和任务定义

当前统一数据集为 VT-Tiny-MOT。

实验分为两个任务：

| 任务 | 检测对象 | 主评价 key | 输出含义 |
| --- | --- | --- | --- |
| 个体级异常检测 | 单条轨迹或单个 track | `sample_id` | 每条个体轨迹一个 anomaly score。 |
| 群体级异常检测 | 群体窗口 | `sample_id + window_id` | 每个群体时间窗口一个 anomaly score。 |

当前实验不是直接使用数据集原始异常标签，而是在统一协议中构造 normal/anomaly 对照。这样做的目的不是声称这是数据集原始 benchmark，而是为了在同一数据、同一异常注入、同一评价指标下比较不同方法。

## 4. 实验协议

### 4.1 验证集协议

validation 协议用于开发、调参和选择方法配置。

当前设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | VT-Tiny-MOT |
| Source split | `train` |
| Sequence validation ratio | `0.2` |
| Main seed | `42` |
| Individual anomaly fraction | `0.1` |
| Group anomaly fraction | `0.1` |
| Group window size / stride | `16 / 8` |
| Individual key | `sample_id` |
| Group key | `sample_id + window_id` |
| Rank metric | `P@100 / R@100` |

validation 协议的用途：

1. 对 FusionTrack 的权重、校准参数和 gate 参数做选择。
2. 对 baseline 适配流程做调试。
3. 发现 key 对齐、score 覆盖和异常注入错误。
4. 生成可复现实验配置，而不是直接作为最终泛化结论。

### 4.2 保留测试集协议

holdout 协议用于最终确认。

当前设置：

| 项目 | 设置 |
| --- | --- |
| Train source split | `train` |
| Eval source split | `test` |
| Seeds | `42, 43, 44` |
| Individual anomaly fraction | `0.1` |
| Group anomaly fraction | `0.1` |
| Group window size / stride | `16 / 8` |
| Individual key | `sample_id` |
| Group key | `sample_id + window_id` |

这个协议的关键点：

1. 所有模型或 scorer 的 normal reference 只能来自 `train`。
2. 异常注入、score 输出和评价发生在 `test`。
3. 不允许用 test 指标继续调参。
4. 如果需要改进方法，必须先在 validation 上固定方案，再重新跑 holdout。

当前 holdout 多种子输出：

```text
/root/autodl-tmp/fusiontrack_holdout_multiseed_combined_20260522/
```

本地归档：

```text
server_artifacts/final_results_20260522/fusiontrack_holdout_multiseed_combined_20260522_summaries.tar.gz
server_artifacts/final_results_20260522/holdout_multiseed_20260522/fusiontrack_holdout_multiseed_combined_20260522/
```

### 4.3 严格键对齐规则

所有主实验必须满足：

1. `num_duplicate_label_keys = 0`
2. `num_duplicate_score_keys = 0`
3. `num_missing_score_keys = 0`
4. `num_extra_score_keys = 0`

个体级：

```text
label key = sample_id
score key = sample_id
```

群体级：

```text
label key = sample_id + window_id
score key = sample_id + window_id
```

早期 `sample_id` any-window 聚合只能作为 appendix diagnostic，不能作为群体级主表。

## 5. 异常注入设计

### 5.1 个体级异常

个体级异常针对单条轨迹进行扰动。当前协议中覆盖的异常类型包括：

| 类型 | 含义 |
| --- | --- |
| `route_shift` | 整体路径偏移，使轨迹偏离正常运动位置。 |
| `speed_spike` | 局部速度突增，制造不自然的瞬时运动。 |
| `stop_or_slowdown` | 局部停止或明显减速。 |
| `jump` | 某些点出现突跳，模拟轨迹定位或运动异常。 |
| `shape_warp` | 轨迹形状变形，改变整体空间模式。 |
| `modal_offset` | 跨模态或坐标相关的整体偏移扰动。 |

这些异常最终都会落到统一的 `sample_id` 级 label 上。

### 5.2 群体级异常

群体级异常针对同一时间窗口内多个目标的关系和群体运动模式。当前协议中覆盖的异常类型包括：

| 类型 | 含义 |
| --- | --- |
| `leave_group` | 个体离开群体，使局部邻域结构异常。 |
| `against_motion` | 个体或子群体逆向运动。 |
| `neighbor_replacement` | 替换邻域成员，破坏正常相邻关系。 |
| `population_change` | 群体规模突变。 |
| `dispersion_change` | 群体扩散或收缩程度异常。 |
| `split_merge` | 群体分裂或合并模式异常。 |

这些异常最终落到 `sample_id + window_id` 级 label 上。

## 6. 统一评价指标

当前所有方法统一输出 anomaly score，score 越高表示越异常。

主指标：

| 指标 | 含义 |
| --- | --- |
| AUROC | 排序区分能力，对正负比例相对不敏感。 |
| AUPRC | 异常稀疏场景更重要，反映高置信异常检出能力。 |
| Best F1 | 在所有阈值上选择最佳 F1，反映可阈值化检测效果。 |
| P@100 | top-100 检出中的异常比例。 |
| R@100 | top-100 覆盖到的异常比例。 |

诊断指标：

| 指标 | 用途 |
| --- | --- |
| `num_label_rows` | label 行数。 |
| `num_score_rows` | score 行数。 |
| `num_unique_label_keys` | 唯一 label key 数。 |
| `num_unique_score_keys` | 唯一 score key 数。 |
| `num_duplicate_label_keys` | 重复 label key，主实验必须为 0。 |
| `num_duplicate_score_keys` | 重复 score key，主实验必须为 0。 |
| `num_missing_score_keys` | label 中存在但 score 缺失的 key，主实验必须为 0。 |
| `num_extra_score_keys` | score 中存在但 label 缺失的 key，主实验必须为 0。 |

## 7. 方法分类

### 7.1 我们的方法：FusionTrack 个体级

#### `fusiontrack_individual_nn`

最基础的 FusionTrack 个体级 nearest-neighbor scorer。它用正常轨迹特征作为 reference，对每条测试轨迹计算最近邻距离，距离越大越异常。

定位：

```text
我们的基础个体级方法 / 非学习式 / no epoch
```

#### `fusiontrack_individual_context`

上下文增强的个体级 scorer。它和普通 nearest-neighbor 的区别在于引入上下文特征，使 score 不只反映轨迹本身形态，也反映局部运动上下文。

定位：

```text
我们的个体级上下文方法 / 非学习式 / no epoch
```

#### `fusiontrack_individual_ensemble`

当前个体级 FusionTrack 主体方法。它融合三类组件：

1. nearest-feature distance
2. LOF novelty score
3. Isolation Forest score

融合方式是 rank ensemble。先把不同组件分数转为 rank，再按权重加权，减少不同 score 尺度不一致的问题。

默认定位：

```text
我们的个体级 ensemble 方法 / 非学习式 / no epoch
```

#### `fusiontrack_individual_ensemble_calibrated`

这是我新增的个体级改进版本，在 `fusiontrack_individual_ensemble` 之后加入 feature-stratified rank calibration。

新增参数：

| 参数 | 含义 |
| --- | --- |
| `calibration_columns` | 用于分层校准的特征列，当前主要包括 `mean_speed`, `duration_frames`, `num_points`。 |
| `calibration_bins` | 每个校准特征的 quantile 分箱数量。 |
| `calibration_global_weight` | 全局 rank 与分层 rank 的混合权重。 |

设计动机：

1. 轨迹长度、速度和点数会影响异常分数分布。
2. 直接全局排序会让某些天然高分的正常轨迹排到前面。
3. 分层 rank calibration 让 score 在相似运动属性内比较，再和全局 rank 混合。

代码位置：

```text
fusiontrack/individual_scoring.py
```

新增核心 helper：

```text
_feature_stratified_rank01
_calibration_metadata
_quantile_strata
_rank_within_strata
```

#### `fusiontrack_individual_ensemble_tuned_auprc`

validation score-grid 选出的个体级 AUPRC 最优候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| nearest-feature | `0.45` |
| LOF novelty | `0.45` |
| Isolation Forest | `0.10` |

校准：

```text
calibration_columns = mean_speed, duration_frames, num_points
calibration_bins = 4
calibration_global_weight = 0.3
```

注意：

这个方法是 validation-tuned candidate，不是手工看 test 调出来的。它可以进入 holdout 对比，但后续不能再用 holdout 指标调整它。

#### `fusiontrack_individual_ensemble_tuned_topk`

validation score-grid 选出的个体级 TopK 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| nearest-feature | `0.60` |
| LOF novelty | `0.30` |
| Isolation Forest | `0.10` |

校准同样使用：

```text
calibration_columns = mean_speed, duration_frames, num_points
calibration_bins = 4
calibration_global_weight = 0.3
```

### 7.2 我们的方法：FusionTrack 群体级

#### `fusiontrack_group_temporal_knn`

群体级基础 KNN 方法。它对群体窗口特征进行标准化，然后计算测试窗口与正常群体窗口之间的近邻距离。

定位：

```text
我们的基础群体级方法 / 非学习式 / no epoch
```

#### `fusiontrack_group_graph`

群体图结构 scorer。它关注群体内部的空间邻接、速度一致性和图结构变化。

当前配置：

```text
k_neighbors = 3
rho_p = 80
rho_v = 20
eta = 0.5
```

定位：

```text
我们的群体图结构方法 / 非学习式 / no epoch
```

#### `fusiontrack_group_hybrid`

当前群体级 FusionTrack 主体方法。它融合三类组件：

1. prediction residual：用线性预测残差刻画群体运动是否可预测。
2. graph cohesion：用群体图结构一致性刻画邻域关系是否异常。
3. temporal profile：用时间窗口统计特征刻画群体运动模式是否偏离正常。

融合方式同样采用 rank fusion，避免不同组件分数尺度不一致。

定位：

```text
我们的群体级 hybrid 方法 / 非学习式 / no epoch
```

#### `fusiontrack_group_hybrid_gated`

这是我新增的群体级改进版本，在 hybrid fusion 中加入 residual gate。

新增参数：

| 参数 | 含义 |
| --- | --- |
| `use_residual_gate` | 是否启用 residual gate。 |
| `residual_gate_power` | gate 强度指数。 |
| `residual_gate_floor` | gate 下限，避免 side evidence 被完全压掉。 |

设计动机：

1. 群体级 graph/temporal side evidence 有时会在 prediction residual 不高时产生假阳性。
2. residual gate 让 side evidence 更多服从主要的运动预测异常。
3. 如果 residual rank 很低，则 graph/temporal 的异常贡献被压制；如果 residual rank 高，则保留或放大侧向证据。

代码位置：

```text
fusiontrack/group_temporal_profile.py
```

新增核心 helper：

```text
_residual_gated_rank_fusion
_residual_side_gates
_rank_array
```

#### `fusiontrack_group_hybrid_tuned_auroc_topk`

validation score-grid 选出的群体级 AUROC / TopK 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| prediction residual | `0.50` |
| graph cohesion | `0.25` |
| temporal profile | `0.25` |

这个版本是 ungated，因为 validation 结果显示当前 gated 版本没有稳定超过 ungated hybrid。

#### `fusiontrack_group_hybrid_tuned_auprc_f1`

validation score-grid 选出的群体级 AUPRC / F1 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| prediction residual | `0.60` |
| graph cohesion | `0.30` |
| temporal profile | `0.10` |

#### `fusiontrack_group_hybrid_tuned_fine_auprc`

细粒度 group weight search 后的 AUPRC 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| prediction residual | `0.47` |
| graph cohesion | `0.41` |
| temporal profile | `0.12` |

#### `fusiontrack_group_hybrid_tuned_fine_topk`

细粒度 group weight search 后的 TopK 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| prediction residual | `0.45` |
| graph cohesion | `0.43` |
| temporal profile | `0.12` |

#### `fusiontrack_group_hybrid_tuned_fine_f1`

细粒度 group weight search 后的 F1 候选。

配置：

| 组件 | 权重 |
| --- | ---: |
| prediction residual | `0.46` |
| graph cohesion | `0.42` |
| temporal profile | `0.12` |

### 7.3 经典基线方法

经典 baseline 不依赖具体论文源码，属于通用异常检测方法，主要用于建立合理参照。

个体级：

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `individual_iforest` | classical / non-deep | Isolation Forest，使用个体轨迹特征。 |
| `individual_lof` | classical / non-deep | Local Outlier Factor novelty-style baseline。 |
| `individual_ocsvm` | classical / non-deep | One-Class SVM。 |

群体级：

| 方法 | 类型 | 说明 |
| --- | --- | --- |
| `group_iforest` | classical / non-deep | Isolation Forest，使用群体窗口特征。 |
| `group_lof` | classical / non-deep | LOF，使用群体窗口特征。 |
| `group_ocsvm` | classical / non-deep | One-Class SVM，使用群体窗口特征。 |
| `group_prediction_linear` | prediction baseline | 线性预测残差 baseline。 |
| `group_temporal_graph_ae_proxy` | proxy / ablation | 轻量图时序 autoencoder proxy，不作为论文官方 baseline。 |

这些 baseline 的意义：

1. 判断 FusionTrack 是否真的超过强传统方法。
2. 防止只和很弱的随机或简单均值 baseline 比较。
3. 暴露任务本身是否被某个简单统计特征解决。

### 7.4 官方论文源码基线方法

论文型 baseline 的规则已经固定：

```text
所有设计论文实验的 baseline，必须以论文中涉及或作者官方给出的源码进行复现。
本地重写、简化实现或 proxy 实现不能放入论文主表。
```

当前规划和已经接入过的官方源码 baseline：

| 方法 | 任务 | 源码要求 | 当前状态 |
| --- | --- | --- | --- |
| CETrajAD | individual | 使用 `ShuruiCao/comp-ensemble-ad` | 已跑过 full-coverage adapter，可作为 `official_cetrajad_fullcoverage` 报告，但必须说明 adapter 和 `no-loss-history`。 |
| LM-TAD | individual | 使用 `jonathankabala/LMTAD` | 已按用户确认的官方仓库纳入规则。 |
| Pi-DPM | individual | 使用 `arunshar/Physics-Informed-Diffusion-Probabilistic-Model` | 官方源码 baseline 候选。 |
| TranAD | individual/group | 官方源码 | 已跑过严格 key 对齐版本，部分深度 run 仍需要 convergence 状态说明。 |
| Anomaly Transformer | individual/group | 官方源码 | 已跑过严格 key 对齐版本，部分 run 为 max-budget-not-converged。 |
| DCdetector | individual/group | 官方源码 | 已跑过严格 key 对齐版本。 |
| CATCH | individual/group | 官方源码 | recent official-source runner 已接入。 |
| CutAddPaste | individual/group | 官方源码 | recent official-source runner 已接入。 |
| TimeMixer | individual/group | 官方源码 | recent official-source runner 已接入。 |
| SensitiveHUE | individual/group | 官方源码 | supplementary candidate；正式写主表前还要确认 peer-reviewed venue/source record。 |

官方 baseline 进入论文主表前必须记录：

1. GitHub URL。
2. commit hash。
3. license。
4. adapter 说明。
5. environment。
6. run manifest。
7. epoch、batch size、learning rate、window size、seed。
8. convergence status。
9. key 对齐诊断。

### 7.5 内部方法、代理方法与消融

以下方法不能冒充对应论文官方 baseline：

| 本地方法名 | 只能放在哪里 | 原因 |
| --- | --- | --- |
| `individual_complementary_cetrajad_proxy` | internal/proxy/ablation | 不是 CETrajAD 官方源码完整复现。 |
| `individual_trajectory_lm_ngram_proxy` | internal/proxy/ablation | 不是 LM-TAD 官方源码完整复现。 |
| `individual_physics_kinematic_proxy` | internal/proxy/ablation | 不是 Pi-DPM 官方源码完整复现。 |
| `group_temporal_graph_ae_proxy` | internal/proxy/ablation | 是本地轻量 proxy，不是指定论文官方实现。 |

这样做的原因：

1. 论文主表必须可追溯到原论文源码，否则容易被质疑 baseline 不公平。
2. proxy 可以作为消融或分析，但不能用论文方法名称包装。
3. 如果一个 proxy 结果很强，也只能说明某类思想有效，不能说明复现了该论文。

## 8. 我已经做过的新增和修改

### 8.1 个体级校准机制

修改文件：

```text
fusiontrack/individual_scoring.py
```

新增内容：

1. `run_individual_fusiontrack_ensemble` 支持 calibration 参数。
2. 新增 feature-stratified rank calibration。
3. 支持按 `mean_speed`、`duration_frames`、`num_points` 等特征做分层 rank。
4. metadata 中记录校准参数，方便后续复现实验。

对应测试：

```text
tests/test_individual_fusiontrack_scoring.py
```

### 8.2 群体级残差门控

修改文件：

```text
fusiontrack/group_temporal_profile.py
```

新增内容：

1. `run_group_fusiontrack_hybrid` 支持 residual gate。
2. 新增 `_residual_gated_rank_fusion`。
3. 新增 `_residual_side_gates`。
4. 新增 `_rank_array`。
5. metadata 中记录 gate 参数。

对应测试：

```text
tests/test_group_temporal_profile.py
```

### 8.3 基准实验矩阵参数传递

修改文件：

```text
runners/run_benchmark_matrix.py
```

新增内容：

1. individual matrix 支持 calibration 参数透传。
2. group matrix 支持 residual gate 参数透传。
3. 新增 `_string_sequence`，用于稳健解析字符串序列配置。

对应测试：

```text
tests/test_run_benchmark_matrix.py
```

### 8.4 验证协议方法注册

修改文件：

```text
runners/prepare_vt_tiny_mot_protocol.py
```

新增方法行：

```text
fusiontrack_individual_ensemble_calibrated
fusiontrack_group_hybrid_gated
fusiontrack_individual_ensemble_tuned_auprc
fusiontrack_individual_ensemble_tuned_topk
fusiontrack_group_hybrid_tuned_auroc_topk
fusiontrack_group_hybrid_tuned_auprc_f1
fusiontrack_group_hybrid_tuned_fine_auprc
fusiontrack_group_hybrid_tuned_fine_topk
fusiontrack_group_hybrid_tuned_fine_f1
```

对应测试：

```text
tests/test_prepare_vt_tiny_mot_protocol.py
```

### 8.5 分数网格运行器

新增文件：

```text
runners/run_fusiontrack_score_grid.py
```

作用：

1. 读取已经缓存的组件分数。
2. 快速搜索 FusionTrack ensemble / hybrid 权重。
3. 避免每组权重都重新跑底层 scorer。
4. 输出候选方法、指标和配置。

使用场景：

1. validation 上挑选权重。
2. 做消融分析。
3. 记录 tuned candidate 的来源。

注意：

这个 runner 只能用于 validation 选择。不能拿 test score-grid 结果反过来调参。

### 8.6 保留测试协议运行器

新增文件：

```text
runners/prepare_vt_tiny_mot_holdout_protocol.py
```

作用：

1. 从 `train` 构建 normal reference。
2. 从 `test` 生成异常注入和评价标签。
3. 将 validation matrix 中的 score path 改写为 holdout split 对应路径。
4. 输出：

```text
individual_test_matrix.json
group_test_matrix.json
protocol_manifest.json
```

### 8.7 保留测试多种子运行器

新增文件：

```text
runners/run_fusiontrack_holdout_multiseed.py
```

作用：

1. 循环 seeds `42,43,44`。
2. 每个 seed 生成 holdout protocol。
3. 每个 seed 分别运行 individual/group matrix。
4. 汇总所有 run 到 `all_runs.csv`。
5. 聚合 mean/std 到 `aggregate.csv`。
6. 按指标输出 `best_by_metric.json`。
7. 写入 `manifest.json` 记录数据路径、seed、source summary 等。

对应测试：

```text
tests/test_holdout_multiseed_runner.py
```

### 8.8 配置文档更新

修改文件：

```text
configs/final_experiment_settings.json
configs/final_experiment_settings.md
```

更新内容：

1. 当前实验协议。
2. fairness rules。
3. convergence policy。
4. local benchmark methods。
5. official paper baselines。
6. server execution 命令。
7. completed reruns。
8. remaining reruns。
9. latest completed holdout multiseed run。

## 9. 运行方式

### 9.1 生成验证集协议

示例：

```bash
python code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_protocol.py \
  --data-root /root/FusionTrack/data/VT-Tiny-MOT \
  --output-root /root/autodl-tmp/fusiontrack_val_strict/protocol \
  --seed 42
```

输出会包括 individual/group label、normal reference、benchmark matrix 和 manifest。

### 9.2 运行基准实验矩阵

个体级示例：

```bash
python code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --matrix /root/autodl-tmp/fusiontrack_val_strict/protocol/individual_val_matrix.json \
  --result-root /root/autodl-tmp/fusiontrack_val_strict/results/individual
```

群体级示例：

```bash
python code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py \
  --matrix /root/autodl-tmp/fusiontrack_val_strict/protocol/group_val_matrix.json \
  --result-root /root/autodl-tmp/fusiontrack_val_strict/results/group
```

### 9.3 运行分数网格搜索

示例：

```bash
python code/anomaly_detection/benchmark/runners/run_fusiontrack_score_grid.py \
  --summary /root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/results/individual/summary.csv \
  --output-root /root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/score_grid_fast
```

score-grid 只用于 validation 选择权重。

### 9.4 运行训练集到测试集的多种子保留测试

示例：

```bash
python code/anomaly_detection/benchmark/runners/run_fusiontrack_holdout_multiseed.py \
  --data-root /root/FusionTrack/data/VT-Tiny-MOT \
  --output-root /root/autodl-tmp/fusiontrack_holdout_multiseed_20260522 \
  --combined-output-root /root/autodl-tmp/fusiontrack_holdout_multiseed_combined_20260522 \
  --seeds 42,43,44 \
  --train-source-split train \
  --eval-source-split test
```

输出：

```text
aggregate.csv
all_runs.csv
best_by_metric.json
manifest.json
```

### 9.5 服务器 tmux/GPU 运行

完整远程实验建议使用 tmux，避免 ssh 断开导致实验中断。

示例：

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

重新连接：

```bash
tmux attach -t fusiontrack_val
```

说明：

1. 非学习式方法不一定占用 GPU。
2. 深度官方 baseline 应显式记录 GPU 名称、显存、epoch、loss history 和 convergence status。
3. 可以并行跑 CPU classical baseline 和 GPU deep baseline，但必须保证输出目录隔离。

## 10. 当前实验结果

### 10.1 验证集：预注册 FusionTrack 改进

结果路径：

```text
/root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/results
```

个体级：

| 方法 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_individual_ensemble` | 0.623752 | 0.153147 | 0.254642 | 0.150000 | 0.180723 |
| `fusiontrack_individual_ensemble_calibrated` | 0.625052 | 0.160261 | 0.280899 | 0.160000 | 0.192771 |
| `individual_lof` | 0.606512 | 0.159437 | 0.246014 | 0.160000 | 0.192771 |

解释：

1. calibration 后个体级 AUROC、AUPRC、F1 都超过未校准 ensemble。
2. validation 上 calibrated AUPRC 略高于 LOF。
3. 该结果说明 feature-stratified rank calibration 有效，但最终仍以 holdout 为准。

群体级：

| 方法 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_group_hybrid` | 0.692176 | 0.081551 | 0.193548 | 0.150000 | 0.170455 |
| `fusiontrack_group_hybrid_gated` | 0.633823 | 0.079872 | 0.186441 | 0.140000 | 0.159091 |
| `group_prediction_linear` | 0.622238 | 0.093898 | 0.162319 | 0.100000 | 0.113636 |

解释：

1. 原始 hybrid 在 AUROC、F1、P@100、R@100 上强于 linear prediction。
2. gated 版本这轮没有超过 ungated hybrid，因此没有作为主推版本。
3. linear prediction 的 AUPRC 在 validation 上较高，但 TopK 和 F1 不占优。

### 10.2 验证集：分数网格调参候选

个体级：

| 方法 | 权重 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_individual_ensemble_tuned_auprc` | nearest/LOF/IForest = 0.45/0.45/0.10 | 0.624326 | 0.166826 | 0.272727 | 0.190000 | 0.228916 |
| `fusiontrack_individual_ensemble_tuned_topk` | nearest/LOF/IForest = 0.60/0.30/0.10 | 0.618915 | 0.162399 | 0.255639 | 0.200000 | 0.240964 |

群体级：

| 方法 | 权重 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_group_hybrid_tuned_auroc_topk` | pred/graph/temp = 0.50/0.25/0.25 | 0.708855 | 0.082023 | 0.198473 | 0.160000 | 0.181818 |
| `fusiontrack_group_hybrid_tuned_auprc_f1` | pred/graph/temp = 0.60/0.30/0.10 | 0.672912 | 0.092200 | 0.215827 | 0.150000 | 0.170455 |
| `fusiontrack_group_hybrid_tuned_fine_auprc` | pred/graph/temp = 0.47/0.41/0.12 | 0.680515 | 0.098513 | 0.215569 | 0.180000 | 0.204545 |
| `fusiontrack_group_hybrid_tuned_fine_topk` | pred/graph/temp = 0.45/0.43/0.12 | 0.679195 | 0.098164 | 0.216867 | 0.190000 | 0.215909 |
| `fusiontrack_group_hybrid_tuned_fine_f1` | pred/graph/temp = 0.46/0.42/0.12 | 0.679892 | 0.097720 | 0.218182 | 0.180000 | 0.204545 |

解释：

1. tuned candidates 是 validation 上选出的候选，不是 test 调参产物。
2. 个体级 tuned AUPRC 和 TopK 分别服务于不同目标。
3. 群体级 fine candidates 提高了 AUPRC、F1 和 TopK，但 AUROC 最强仍是 `tuned_auroc_topk`。

### 10.3 保留测试：训练集到测试集，种子 42/43/44

结果路径：

```text
server_artifacts/final_results_20260522/holdout_multiseed_20260522/
  fusiontrack_holdout_multiseed_combined_20260522/aggregate.csv
```

严格对齐状态：

```text
78/78 metric rows
duplicate label keys = 0
duplicate score keys = 0
missing score keys = 0
extra score keys = 0
```

个体级结果：

| 类别 | 方法 | AUROC mean±std | AUPRC mean±std | F1 mean±std | P@100 mean±std | R@100 mean±std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 我们的方法 | `fusiontrack_individual_ensemble` | 0.636716±0.020666 | 0.176854±0.015940 | 0.254073±0.020550 | 0.210000±0.010000 | 0.131250±0.006250 |
| 我们的方法 | `fusiontrack_individual_ensemble_calibrated` | 0.643727±0.025806 | 0.181702±0.018038 | 0.263375±0.022913 | 0.216667±0.011547 | 0.135417±0.007217 |
| 我们的方法 | `fusiontrack_individual_ensemble_tuned_auprc` | 0.645935±0.031040 | 0.190092±0.021638 | 0.274699±0.031964 | 0.246667±0.020817 | 0.154167±0.013010 |
| 我们的方法 | `fusiontrack_individual_ensemble_tuned_topk` | 0.641182±0.026501 | 0.180961±0.020149 | 0.267826±0.025213 | 0.233333±0.020817 | 0.145833±0.013010 |
| 我们的方法 | `fusiontrack_individual_nn` | 0.606146±0.013770 | 0.152517±0.014666 | 0.251243±0.031687 | 0.153333±0.011547 | 0.095833±0.007217 |
| 我们的方法 | `fusiontrack_individual_context` | 0.602402±0.008573 | 0.147572±0.009708 | 0.241643±0.003995 | 0.156667±0.011547 | 0.097917±0.007217 |
| 经典 baseline | `individual_iforest` | 0.610901±0.010634 | 0.137805±0.005914 | 0.241162±0.010640 | 0.140000±0.017321 | 0.087500±0.010825 |
| 经典 baseline | `individual_lof` | 0.634865±0.034705 | 0.191153±0.042912 | 0.258759±0.023014 | 0.223333±0.055076 | 0.139583±0.034422 |
| 经典 baseline | `individual_ocsvm` | 0.521611±0.006921 | 0.116739±0.007962 | 0.189130±0.005404 | 0.156667±0.011547 | 0.097917±0.007217 |
| proxy/ablation | `individual_complementary_cetrajad_proxy` | 0.574390±0.010746 | 0.127538±0.005332 | 0.210743±0.008276 | 0.126667±0.005774 | 0.079167±0.003608 |
| proxy/ablation | `individual_physics_kinematic_proxy` | 0.613265±0.003856 | 0.144495±0.007726 | 0.256121±0.013726 | 0.120000±0.030000 | 0.075000±0.018750 |
| proxy/ablation | `individual_trajectory_lm_ngram_proxy` | 0.445483±0.015583 | 0.093928±0.003189 | 0.184444±0.002342 | 0.146667±0.037859 | 0.091667±0.023662 |

个体级解读：

1. `fusiontrack_individual_ensemble_tuned_auprc` 是当前我们方法中最强版本。
2. 它在 AUROC、F1、P@100、R@100 上超过所有当前个体级方法和经典 baseline。
3. `individual_lof` 的 AUPRC 为 `0.191153±0.042912`，略高于我们的 `0.190092±0.021638`。
4. 由于差距很小但确实存在，论文中不能声称个体级 AUPRC 已经完全最好。下一步如果要优化，必须在 validation 上提出新方案，再重新 holdout。

群体级结果：

| 类别 | 方法 | AUROC mean±std | AUPRC mean±std | F1 mean±std | P@100 mean±std | R@100 mean±std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 我们的方法 | `fusiontrack_group_hybrid` | 0.757597±0.013936 | 0.083009±0.020258 | 0.166737±0.023919 | 0.190000±0.030000 | 0.124183±0.019608 |
| 我们的方法 | `fusiontrack_group_hybrid_gated` | 0.669373±0.012332 | 0.078897±0.015942 | 0.173458±0.026985 | 0.203333±0.030551 | 0.132898±0.019968 |
| 我们的方法 | `fusiontrack_group_hybrid_tuned_auroc_topk` | 0.794720±0.018227 | 0.085337±0.021904 | 0.169019±0.023337 | 0.190000±0.043589 | 0.124183±0.028490 |
| 我们的方法 | `fusiontrack_group_hybrid_tuned_auprc_f1` | 0.748898±0.011626 | 0.086318±0.016951 | 0.151657±0.017892 | 0.180000±0.030000 | 0.117647±0.019608 |
| 我们的方法 | `fusiontrack_group_hybrid_tuned_fine_auprc` | 0.785322±0.019491 | 0.090821±0.020840 | 0.153566±0.025713 | 0.180000±0.026458 | 0.117647±0.017292 |
| 我们的方法 | `fusiontrack_group_hybrid_tuned_fine_topk` | 0.787700±0.020840 | 0.091499±0.021386 | 0.154654±0.025797 | 0.180000±0.036056 | 0.117647±0.023566 |
| 我们的方法 | `fusiontrack_group_hybrid_tuned_fine_f1` | 0.786590±0.020286 | 0.091221±0.020809 | 0.154082±0.025810 | 0.176667±0.030551 | 0.115468±0.019968 |
| 我们的方法 | `fusiontrack_group_temporal_knn` | 0.217411±0.027677 | 0.004083±0.000296 | 0.025160±0.003487 | 0.010000±0.010000 | 0.006536±0.006536 |
| 我们的方法 | `fusiontrack_group_graph` | 0.270917±0.014351 | 0.004432±0.000109 | 0.013875±0.004814 | 0.006667±0.005774 | 0.004357±0.003774 |
| 经典 baseline | `group_iforest` | 0.188697±0.011579 | 0.003245±0.000288 | 0.011197±0.000800 | 0.003333±0.005774 | 0.002179±0.003774 |
| 经典 baseline | `group_lof` | 0.296523±0.020502 | 0.003461±0.000093 | 0.010754±0.000001 | 0.000000±0.000000 | 0.000000±0.000000 |
| 经典 baseline | `group_ocsvm` | 0.297740±0.026193 | 0.004438±0.000466 | 0.026717±0.007399 | 0.010000±0.010000 | 0.006536±0.006536 |
| 经典 baseline | `group_prediction_linear` | 0.637508±0.008517 | 0.017417±0.003255 | 0.071098±0.001315 | 0.020000±0.010000 | 0.013072±0.006536 |
| proxy/ablation | `group_temporal_graph_ae_proxy` | 0.365516±0.030061 | 0.004639±0.000567 | 0.021820±0.012760 | 0.016667±0.005774 | 0.010893±0.003774 |

群体级解读：

1. `fusiontrack_group_hybrid_tuned_auroc_topk` 当前 AUROC 最强：`0.794720±0.018227`。
2. `fusiontrack_group_hybrid_tuned_fine_topk` 当前 AUPRC 最强：`0.091499±0.021386`。
3. 所有主要 FusionTrack hybrid 版本都明显高于 `group_prediction_linear`。
4. 基础 graph 和 temporal KNN 单独使用效果很弱，说明群体任务依赖多组件融合。

## 11. 官方基线方法当前说明

已经完成或记录过的官方 baseline rerun 包括：

1. LM-TAD、Pi-DPM、TranAD、Anomaly Transformer、DCdetector 的严格 key 对齐 run。
2. CETrajAD 原始 adapter run，发现 `770/829` score rows，缺失 `59` 个 score keys。
3. CETrajAD full-coverage adapter run，修复到 `829/829`，可作为 `official_cetrajad_fullcoverage` 报告。
4. CATCH、CutAddPaste、TimeMixer、SensitiveHUE recent official-source runner 接入。
5. 深度 baseline 的 50-epoch max-budget rerun，其中部分方法仍是 `max-budget-not-converged`。

CETrajAD full-coverage adapter 当前结果：

| 方法 | AUROC | AUPRC | F1 | P@100 | R@100 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_cetrajad_fullcoverage` | 0.521092 | 0.106465 | 0.193437 | 0.080000 | 0.096386 | `no-loss-history` |

论文写法注意：

1. 不能把失败的 `770/829` 原始 adapter 结果放主表。
2. 可以报告 full-coverage adapter，但必须说明 adapter 做了覆盖修复、`coordinate_scale=1.0`、且没有 loss history。
3. 深度 baseline 如果 loss 仍在下降，不能声称 fully converged，只能标注 `max-budget-not-converged` 或继续延长训练。

## 12. 收敛和训练轮次规则

不同算法不强制 epoch 完全一样，因为：

1. LOF、IForest、OCSVM、KNN、rank fusion 这类方法没有 epoch。
2. 不同深度模型的训练机制、window、loss 和 early stopping 条件不同。
3. 强制相同 epoch 可能反而不公平。

统一规则是：

1. 每个方法必须预先声明 training budget。
2. 每个深度方法必须记录 epoch、batch size、learning rate、window size、seed。
3. 支持 validation loss 的方法使用 early stopping。
4. 不支持 validation loss 的方法至少记录 training loss 和 score-stability diagnostic。
5. 50 epoch 只是初始最大预算，不代表一定收敛。

最终论文表可接受的 convergence status：

| 状态 | 含义 |
| --- | --- |
| `converged` | loss 按规则 plateau。 |
| `early-stopped` | validation loss 触发 early stopping 并恢复 best checkpoint。 |
| `max-budget-not-converged` | 到预算上限还在改善，不能声称完全收敛。 |
| `no-loss-history` | 官方适配或非标准 runner 没有 loss history，必须明确报告限制。 |
| `no-epoch` | 非学习式方法，无 epoch。 |

## 13. GitHub 提交边界

可以提交到项目 GitHub 的内容：

1. benchmark runner。
2. FusionTrack scorer 代码。
3. protocol 生成脚本。
4. score-grid 脚本。
5. holdout/multiseed 聚合脚本。
6. adapter 代码。
7. 测试代码。
8. README 和实验配置文档。
9. 小型 summary、manifest 或结果表，如果不包含隐私、凭据、大文件和原始数据。

不应该提交的内容：

1. SSH 密码、token、API key、服务器登录信息。
2. 原始数据集。
3. 大型 score 文件、checkpoint、模型权重、完整 logs。
4. 第三方官方源码的完整拷贝，除非 license 和 submodule/vendor policy 已明确。
5. 临时 tmux 输出、cache、中间 pickle、临时压缩包。
6. 任何包含服务器敏感路径或凭据的文件。

第三方论文源码建议方式：

1. 使用外部 checkout。
2. 记录 GitHub URL 和 commit。
3. 写 adapter 调用它。
4. 不直接 vendor 进本仓库，除非 license 允许且项目明确需要。

## 14. 验证命令

文档对应代码修改已通过过以下验证：

```bash
python -m py_compile \
  code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_holdout_protocol.py \
  code/anomaly_detection/benchmark/runners/run_fusiontrack_holdout_multiseed.py \
  code/anomaly_detection/benchmark/runners/run_fusiontrack_score_grid.py
```

```bash
pytest code/anomaly_detection/benchmark/tests
```

最近一次完整测试结果：

```text
26 passed, 1 warning
```

也做过 diff 格式检查：

```bash
git diff --check
```

结果只有 Windows 环境下的 LF -> CRLF warning，没有 whitespace error。

## 15. 当前仍未完成的事情

### 15.1 个体级方法继续优化

当前我们方法在 holdout 的 AUPRC 上略低于 LOF：

```text
individual_lof AUPRC = 0.191153±0.042912
fusiontrack_individual_ensemble_tuned_auprc AUPRC = 0.190092±0.021638
```

后续可做：

1. 在 validation 上设计新的 calibration。
2. 尝试 anomaly-type-aware 但不使用 test label 的稳定特征。
3. 加入 score uncertainty 或 robust rank aggregation。
4. 只在 validation 上选择方案，然后重新 train -> test holdout。

不能做：

```text
直接看 test 结果再调权重。
```

### 15.2 官方论文基线方法收敛补跑

还需要对 `max-budget-not-converged` 的深度方法延长训练或明确报告状态。

优先项：

1. Anomaly Transformer。
2. TranAD。
3. 其它 recent official-source baselines。

每个补跑都要保存：

```text
loss_history.json
best_epoch
final_epoch
early_stop_reason
GPU name
wall time
run_manifest.json
```

### 15.3 官方基线方法保留测试多种子

当前 holdout 多种子汇总主要覆盖 local/proxy/FusionTrack 方法。论文主表如果要完整，应把官方源码 baseline 也按同一 train -> test seeds 42/43/44 协议跑完。

要求：

1. 使用官方源码。
2. 使用同一异常注入协议。
3. 使用同一 key 对齐规则。
4. 使用同一 AUROC/AUPRC/F1/P@100/R@100。
5. 保存 official run manifest。

### 15.4 论文结果表整理

建议最终拆成四张表：

1. 个体级主表：我们的 FusionTrack vs 经典 baseline vs 官方源码 baseline。
2. 群体级主表：我们的 FusionTrack vs 经典 baseline vs 官方源码 baseline。
3. 消融表：ensemble、calibration、gate、不同权重、单组件。
4. appendix/proxy 表：所有 proxy、失败 adapter、any-window diagnostic。

## 16. 推荐论文表述

对当前结果可以这样表述：

1. FusionTrack 在群体级异常检测上表现稳定，hybrid 系列在 AUROC、AUPRC 和 TopK 上显著优于经典群体 baseline。
2. 个体级 FusionTrack tuned ensemble 在多数指标上超过经典 baseline，但 AUPRC 与 LOF 非常接近，目前 LOF 仍有极小优势。
3. feature-stratified calibration 在 validation 和 holdout 上都提升了个体级 ensemble。
4. residual gate 作为一个消融项有意义，但当前不是主推群体级配置。
5. 所有论文型 baseline 均应以官方源码复现结果为准，本地 proxy 只作为补充分析。

## 17. 最重要的复现实验产物

当前最值得保留和引用的文件：

```text
code/anomaly_detection/benchmark/configs/final_experiment_settings.md
code/anomaly_detection/benchmark/configs/final_experiment_settings.json
code/anomaly_detection/benchmark/runners/prepare_vt_tiny_mot_holdout_protocol.py
code/anomaly_detection/benchmark/runners/run_fusiontrack_holdout_multiseed.py
code/anomaly_detection/benchmark/runners/run_fusiontrack_score_grid.py
server_artifacts/final_results_20260522/holdout_multiseed_20260522/fusiontrack_holdout_multiseed_combined_20260522/aggregate.csv
server_artifacts/final_results_20260522/holdout_multiseed_20260522/fusiontrack_holdout_multiseed_combined_20260522/all_runs.csv
server_artifacts/final_results_20260522/holdout_multiseed_20260522/fusiontrack_holdout_multiseed_combined_20260522/best_by_metric.json
server_artifacts/final_results_20260522/holdout_multiseed_20260522/fusiontrack_holdout_multiseed_combined_20260522/manifest.json
```

如果后续继续实验，应先检查这些文件，再决定是否需要重新生成协议或重新跑 benchmark。

## 方法注册表与运行 manifest

从当前版本开始，方法画像统一放在 `configs/method_registry.json`。这个文件是系统里
`owner`、`role`、`method_family`、`learning_type`、`source_type` 和 `status`
这些字段的统一来源，覆盖 individual、group 和 registration 三类任务。

这样做的目的有三个：

1. benchmark runner 生成 `manifest.json` 时，每个 run 都会写入 `method_profile`，
   后续汇总、论文表格和页面展示不需要再猜测方法归属。
2. 最终可视化看板在缺少 `final_*_all_methods_categorized.csv` 时，会自动回退到同一个
   registry，避免算法接入页面出现空的 owner/family/learning 字段。
3. 后续新增方法时，应优先在 `configs/method_registry.json` 注册方法画像，再把方法
   加入 matrix 配置或 runner。不要只在前端或结果 CSV 里临时写分类字段。

如果某个实验 run 的名字不是 registry 里的 canonical name，可以在 registry 条目中加入
`aliases`，或者在 matrix 实验项里显式设置 `method_registry_name`。manifest 会保留原始
run name，同时写入 registry 的 canonical `method_profile.name`。

## 2026-05-25 更新：labels/scores schema 校验

本轮新增 `evaluation/schema.py`，把评估输入的结构约束从指标计算逻辑里独立出来，作为 benchmark 的统一治理入口。

当前校验规则如下：

1. label 行必须包含任务 key 字段，默认 individual 使用 `sample_id`，group 使用 `sample_id + window_id`。
2. label 行必须包含二值 `label`，只允许 `0/1`。
3. score 行必须包含任务 key 字段和有限数值 `score`，`NaN`、空值、缺失字段会直接失败。
4. 如果同时存在 `frame_start` 和 `frame_end`，必须满足 `frame_end >= frame_start`。
5. 当 CLI 或 reporting 入口启用 `--require-unique-keys` 时，重复 label key 或 score key 会直接失败。

该校验已经接入 `evaluation.reporting.evaluate_score_file()`，因此 `run_evaluation.py`、批量矩阵评估和后续 dashboard 结果聚合都会在指标计算前先检查输入数据。这样可以避免错误 score 被 `_finite_scores()` 或对齐逻辑静默吞掉，保证实验表格中的 AUROC/AUPRC/F1/P@K/R@K 都来自结构合法的数据文件。

## 2026-05-25 更新：benchmark run manifest

`runners/run_benchmark_matrix.py` 生成的 `manifest.json` 已升级到 `manifest_schema_version = 2`。除原有的 config、label、summary、method registry 和 runs 信息外，现在会额外记录：

1. `generated_at_utc`：本次矩阵运行的 UTC 时间。
2. `config_sha256`：矩阵配置文件的 SHA-256，用于确认结果对应的配置版本。
3. `git`：当前仓库 commit、branch 和 dirty 状态，用于判断结果是否来自干净版本。
4. `environment`：Python 版本和平台信息。
5. `inputs`：label 文件、key 字段和 P@K 使用的 `k`。
6. 每个 run 的 `experiment_config`、`score_sha256` 和 `metrics_sha256`。

这一步把“跑出结果”推进为“结果可追溯”：后续无论是在本地、服务器还是导出包中查看结果，都能判断该结果对应的代码版本、配置哈希、输入约束和每个方法的实际运行参数。

## 2026-05-25 更新：holdout multiseed manifest

`runners/run_fusiontrack_holdout_multiseed.py` 现在也会输出 `manifest_schema_version = 2` 的 `manifest.json`。该文件用于记录多种子 holdout 聚合结果的整体来源，包含：

1. 多种子列表、任务层级、train/eval split、异常注入比例、窗口大小和 stride。
2. `all_runs.csv`、`aggregate.csv`、`best_by_metric.json` 三个核心聚合文件的路径与 SHA-256。
3. 当前 git commit、branch、dirty 状态和 Python 运行环境。
4. 兼容旧字段的 `all_runs_csv`、`aggregate_csv`、`best_by_metric_json`，保证已有脚本仍可读取。

这样 validation 矩阵和最终 holdout 多种子聚合都具备基础可追溯字段。后续官方 baseline runner 接入时也应复用同类字段，避免只保留分散日志而无法追溯最终论文表格。
