# 2026-05-22 实验审计记录

本文档记录本轮除论文主表写作外已经完成的工作、结果来源和剩余风险。

## 完成状态

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 远程结果归档 | 已完成 | `/root/autodl-tmp/fusiontrack_final_results_20260522.tar.gz` 和 `/root/autodl-tmp/fusiontrack_update_results_20260522.tar.gz` |
| 官方 deep baseline 长预算复跑 | 已完成 | `/root/autodl-tmp/fusiontrack_b3b8599_convergence_20260522` |
| 我们方法增强 | 已完成 | `fusiontrack_individual_ensemble`、`fusiontrack_group_hybrid` 已接入 runner 和默认 protocol matrix |
| strict validation 复跑 | 已完成 | `/root/autodl-tmp/fusiontrack_b3b8599_methods_20260522` |
| 本地审计归档 | 已完成 | `server_artifacts/final_results_20260522/fusiontrack_update_results_20260522.tar.gz` |
| 论文主表写作 | 已跳过 | 按用户要求，本轮不做第 3 项 |
| GitHub 推送 | 未完成 | 本地分支提交存在，但远程 HTTPS 连接 reset/timeout，需网络恢复后重试 |

## 规则更新

已在 `code/anomaly_detection/benchmark/policies/paper_source_reproduction_policy.md` 写入强制规则：所有涉及论文方法的正式实验必须优先使用论文官方源码或论文明确指向的源码复现。不能用本地 proxy 冒充原论文方法。

## 我们的方法

| 方法 | 任务 | 学习类型 | 说明 |
| --- | --- | --- | --- |
| `fusiontrack_individual_nn` | individual | 学习型 | 手工轨迹特征 nearest-neighbor profile |
| `fusiontrack_individual_ensemble` | individual | 学习型 | nearest-feature、LOF novelty、Isolation Forest 的无标签 rank ensemble |
| `fusiontrack_individual_context` | individual | 学习型 | 加入群体上下文特征的 nearest-neighbor profile |
| `fusiontrack_group_graph` | group | 非学习型 | 图关系、相对运动和群体事件规则打分 |
| `fusiontrack_group_temporal_knn` | group | 学习型 | 群体窗口特征 KNN |
| `fusiontrack_group_hybrid` | group | 学习型/融合型 | prediction residual、graph cohesion、temporal profile 的 rank fusion |

## 经典 Baseline

| 方法 | 任务 | 学习类型 | 说明 |
| --- | --- | --- | --- |
| `individual_lof` | individual | 学习型 | 经典 LOF novelty baseline |
| `individual_iforest` | individual | 学习型 | Isolation Forest |
| `individual_ocsvm` | individual | 学习型 | One-Class SVM |
| `group_prediction_linear` | group | 非学习型 | 线性运动预测残差 |
| `group_lof` | group | 学习型 | 群体特征 LOF |
| `group_iforest` | group | 学习型 | 群体特征 Isolation Forest |
| `group_ocsvm` | group | 学习型 | 群体特征 One-Class SVM |

## 官方论文 Baseline

| 方法 | 任务 | 来源要求 | 本轮状态 |
| --- | --- | --- | --- |
| LM-TAD | individual | `jonathankabala/LMTAD` 官方源码 | 50 epoch 长预算已收敛 |
| Pi-DPM | individual | 论文源码 | 原 20 epoch run 已收敛，本轮未重跑 |
| TranAD | individual/group | `imperial-qore/TranAD` 官方源码 | 50 epoch 后仍 `max-budget-not-converged` |
| Anomaly Transformer | individual/group | 官方源码 | 50 epoch 后仍 `max-budget-not-converged` |
| DCdetector | individual/group | 官方源码 | 原 8 epoch run 已收敛，本轮未重跑 |
| CETrajAD | individual | `ShuruiCao/comp-ensemble-ad` 官方源码 | coverage failed，770/829 score，暂不进 strict 主表 |

## Enhanced Method 结果

远程路径：`/root/autodl-tmp/fusiontrack_b3b8599_methods_20260522`。

### Individual

| 方法 | 类别 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_individual_ensemble` | 我们的方法 | 0.623752 | 0.153147 | 0.254642 | 0.150000 | 0.180723 |
| `individual_lof` | 经典 baseline | 0.606512 | 0.159437 | 0.246014 | 0.160000 | 0.192771 |
| `fusiontrack_individual_nn` | 我们的方法 | 0.595626 | 0.134131 | 0.237681 | 0.180000 | 0.216867 |
| `individual_iforest` | 经典 baseline | 0.592445 | 0.127709 | 0.218310 | 0.160000 | 0.192771 |

结论：`fusiontrack_individual_ensemble` 当前取得最高 AUROC 和 F1，但 AUPRC、P@100、R@100 仍低于 `individual_lof` 或 `fusiontrack_individual_nn`，最终论文表述应按指标分别说明。

### Group

| 方法 | 类别 | AUROC | AUPRC | F1 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fusiontrack_group_hybrid` | 我们的方法 | 0.692177 | 0.081551 | 0.193548 | 0.150000 | 0.170455 |
| `group_prediction_linear` | 经典 baseline | 0.622238 | 0.093898 | 0.162319 | 0.100000 | 0.113636 |
| `group_lof` | 经典 baseline | 0.417786 | 0.005899 | 0.031332 | 0.020000 | 0.022727 |
| `fusiontrack_group_graph` | 我们的方法 | 0.361618 | 0.006877 | 0.041152 | 0.030000 | 0.034091 |

结论：`fusiontrack_group_hybrid` 当前取得最高 AUROC、F1、P@100、R@100；AUPRC 仍低于 `group_prediction_linear`。该方法使用固定配置中的 inverse-rank graph/temporal components，需在 test split 或多 seed 上确认后再作为最终论文 claim。

## Long-Budget 官方结果

远程路径：`/root/autodl-tmp/fusiontrack_b3b8599_convergence_20260522`。

| 方法 | 任务 | AUROC | AUPRC | F1 | 收敛状态 |
| --- | --- | ---: | ---: | ---: | --- |
| `official_lmtad_50` | individual | 0.474708 | 0.100031 | 0.183374 | `converged` |
| `official_tranad_50` | individual | 0.455926 | 0.103066 | 0.187283 | `max-budget-not-converged` |
| `official_anomaly_transformer_50` | individual | 0.530217 | 0.110745 | 0.191740 | `max-budget-not-converged` |
| `official_tranad_50` | group | 0.410594 | 0.004745 | 0.013699 | `max-budget-not-converged` |
| `official_anomaly_transformer_50` | group | 0.575085 | 0.013800 | 0.029268 | `max-budget-not-converged` |

## 剩余风险

1. `fusiontrack_group_hybrid` 的 rank direction 是本轮根据方法审查后固定在配置中的增强项，不能把它描述成已在 test split 上最终定型。
2. 还有 4 个官方 deep baseline 在 50 epoch 后未收敛；如果论文要求“完全收敛结果”，需要继续扩展预算。
3. CETrajAD 仍是 coverage failed，不能进入 strict 主表。
4. 当前 strict 结果是 seed 42 validation protocol；最终论文建议补 test split 或多 seed。

## Recent Official Baseline 补充

远程路径：`/root/autodl-tmp/fusiontrack_recent_official_20260522`。
本地归档：`server_artifacts/final_results_20260522/fusiontrack_recent_official_20260522.tar.gz`。

新增 strict official-source 结果均已满足：无重复 label key、无重复 score key、无 missing score key、无 extra score key。个体级使用 `sample_id`，群体级使用 `sample_id + window_id`。

### Individual Recent Official

| 方法 | 类别 | AUROC | AUPRC | F1 | P@100 | R@100 | 收敛状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_catch_individual_20` | official/top-venue | 0.543759 | 0.124095 | 0.204793 | 0.120000 | 0.144578 | `max-budget-not-converged` |
| `official_timemixer_individual_20` | official/top-venue | 0.521658 | 0.113224 | 0.197441 | 0.090000 | 0.108434 | `converged` |
| `official_cutaddpaste_individual_20` | official/top-venue | 0.472916 | 0.128043 | 0.182222 | 0.090000 | 0.108434 | `max-budget-not-converged` |
| `official_sensitive_hue_individual_20` | supplementary official-source | 0.524387 | 0.110813 | 0.196796 | 0.100000 | 0.120482 | `max-budget-not-converged` |

### Group Recent Official

| 方法 | 类别 | AUROC | AUPRC | F1 | P@100 | R@100 | 收敛状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_timemixer_group_20` | official/top-venue | 0.627657 | 0.014647 | 0.033520 | 0.030000 | 0.034091 | `max-budget-not-converged` |
| `official_catch_group_20` | official/top-venue | 0.605357 | 0.019403 | 0.022472 | 0.010000 | 0.011364 | `max-budget-not-converged` |
| `official_cutaddpaste_group_20` | official/top-venue | 0.561328 | 0.020299 | 0.058252 | 0.040000 | 0.045455 | `max-budget-not-converged` |
| `official_sensitive_hue_group_20` | supplementary official-source | 0.467138 | 0.020232 | 0.046823 | 0.030000 | 0.034091 | `max-budget-not-converged` |

结论：新增顶会 official baseline 中，个体级 AUROC 最好的是 CATCH，群体级 AUROC 最好的是 TimeMixer；它们仍低于当前 FusionTrack 最优个体/群体方法。SensitiveHUE 暂只作为补充结果，因为当前 public README 仍标为 under review。
