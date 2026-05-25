# FusionTrack 系统模块完成度清单（当前版）

## 1. 总体说明

本清单对齐论文目标与当前工程现状，按“模块是否可独立运行”与“端到端可复现闭环”两层给出判断。  
状态定义：
- ✅ 已完成：可直接运行且稳定集成
- 🟡 部分完成：核心能力存在，但未闭环或缺关键治理项
- ☐ 未完成：框架有雏形但未接入主流程

## 2. 模块完成度

### A. 数据与实验配置层

- 数据版本治理：✅（基础闭环）
  现状：系统入口会生成 `dataset_manifest_<split>.json`，记录 annotation hash、图像目录计数、数据集指纹，并写入 pipeline summary / manifest / 导出包。
  下一步：继续把 synthetic protocol 参数与 dataset manifest 绑定成完整 protocol manifest。

- labels/scores schema 与校验：✅（基础闭环）
  现状：`evaluation/schema.py` 已统一校验 label/score 行，评估前会 fail-fast 检查 key、二值标签、有限 score 和帧范围。
  下一步：后续扩展真实数据集 preset 时继续补充字段级提示信息。

- strict key 管理：✅（基础闭环）
  现状：`run_evaluation.py` 和 matrix 配置均支持任务级 `key_fields`；主协议 individual 使用 `sample_id`，group 使用 `sample_id + window_id`，并支持唯一键和 score/label 完全匹配检查。
  下一步：后续把 key policy 也展示到 dashboard/export 的治理说明中。

- 合成异常协议治理：✅（主协议闭环）
  现状：异常注入脚本已输出 manifest v2，validation/holdout 协议生成器会自动生成 `dataset_manifest.json` 并传入注入 manifest，记录参数、文件 hash、label 分布、重放命令和 dataset fingerprint。
  下一步：把协议 manifest 汇总到最终 dashboard/export 的解释层。

- 真实标签并行接口：✅（基础入口）
  现状：新增 `prepare_real_labels.py` 和 `dataset_adapters/real_labels.py`，可把外部真实标注 CSV/JSONL 归一化到统一 label schema，支持 individual 的 `sample_id` 与 group 的 `sample_id + window_id`。
  下一步：后续如果拿到真实异常标注，需要补数据集专属字段映射 preset 和人工标注质量审计。

### B. 融合与轨迹构建层

- 多模态标准化：✅（基础闭环）
  现状：新增 `mtf_ba/observation_standardization.py`，统一把扁平 `observations_<split>.csv` 行转换为 `rgb/thermal/modal_relation/quality` 结构；individual 轨迹导出与 group window 导出已共用该入口，避免两条链路各自解析字段。
  下一步：继续把标准化质量统计汇总到 protocol manifest 与 dashboard 数据流面板。

- 跨模态关联与跨帧关联：✅（基础闭环）
  现状：新增 `mtf_ba/fused_track_pipeline.py` 与 `export_fused_track_pipeline.py`，可从 `observations_<split>.csv` 一次生成 individual trajectories、fused trajectories、group windows、summary 和 manifest；fused 轨迹点保留跨模态中心融合、来源模态、offset confidence，轨迹级保留 frame ids/gaps/max gap 等跨帧 linkage。
  下一步：把该 pipeline 作为 protocol 生成器的默认内部路径，减少旧脚本分步调用。

- 噪声抑制与目标持久化策略：✅（基础闭环）
  现状：`TrackQualityConfig` 已接入 `fused track pipeline`，支持最小轨迹点数、最小可见帧、最大帧间断裂、最小 fused ratio 和是否保留过滤轨迹；每条 fused trajectory 写入 `quality.keep/drop_reasons`，summary/manifest 记录策略参数与过滤原因计数，group windows 会同步移除被过滤目标。
  下一步：根据真实实验分布在 validation 上固定默认阈值，并把过滤统计展示到 dashboard 数据流面板。

- 融合轨迹可追溯输出：✅（基础闭环）
  现状：`fused track pipeline` 已统一输出目录与文件命名，并写出 `fused_track_pipeline_summary_<split>.json` 和 `fused_track_pipeline_manifest_<split>.json`，记录输入 CSV hash、输出 artifact hash、配置和模态覆盖率。
  下一步：把 pipeline manifest 汇总到最终 dashboard/export 的数据治理说明层。

### C. 行为异常检测层

- Individual 分支（route/speed/shape）：✅（基础闭环）
  现状：`fusiontrack/individual_scoring.py` 已为 nearest 与 ensemble score rows 统一输出 `route_score`、`speed_score`、`speed_slowdown_score`、`jump_score`、`shape_score`、`route_shape_score` 和 `modal_offset_score`；这些字段写入 `component_scores`，dashboard 子模块解释面板可直接读取，metadata 记录 schema 版本和分量对应的 feature columns。
  下一步：后续可把分量阈值和 top reason 文案进一步从前端规则迁移为后端事件解释输出。

- Group 分支（群体结构与事件）：✅（基础闭环）
  现状：`fusiontrack/group_scoring.py` 已输出 `event_score`、`event_segments`、`frame_event_scores` 与群体图结构分量；`fusiontrack/group_temporal_profile.py` 已在 `fusiontrack_group_hybrid` score rows 中透传 `graph_leave`、`graph_motion`、`graph_neighbor`、`graph_count`、`graph_dispersion`、`graph_split_merge`、`graph_object_group`、`graph_group_event`，并通过 `metadata.score_sources` 记录 prediction / graph / temporal_profile 三个子分支的原始分数和组件来源。
  下一步：后续把 group top reason 文案和事件段阈值策略进一步固化为后端 explanation schema。

- 个体-群体融合：✅（基础闭环）
  现状：`code/system/fusiontrack/score_fusion.py` 已把 individual/group score rows 融合为统一 JSONL/CSV，输出 `S_ind`、`S_grp`、`S_event`、`S_fused`，并合并 `event_segments` 与 `frame_event_scores`；`final_dashboard.py` 已在 `task_score_decomposition` 中读取顶层 `used_sources`、`metadata.used_sources` 和显式 `component_scores.S_*`，确保网页分数分解条与导出 score JSONL 使用同一条融合解释链。
  下一步：后续可把融合权重、事件阈值和 top reason 策略提升为统一 explanation schema。

- 事件段生成与平滑：☐  
  现状：未在主展示面板形成完整事件段语义。  
  下一步：实现 frame 级 score 序列与事件段合并输出。

### D. 评测与治理层

- 方法注册表：✅（基础闭环）
  现状：`configs/method_registry.json` 已作为 benchmark manifest、最终 dashboard 和方法画像字段的统一来源，支持 aliases 与未注册方法标记。
  下一步：后续新增方法时继续先注册画像，再接入 matrix 或 dashboard。

- 指标聚合：✅（部分）  
  现状：AUROC/AUPRC/F1/P@K/R@K 已支持。  
  下一步：补缺失字段告警与一致性约束。

- 多任务多方法批量评测：✅（基础闭环）
  现状：新增 `run_suite.py`，可一次调度多个 matrix config，分别运行 `run_benchmark_matrix.py`，并输出 `suite_manifest.json` 与 `aggregate_summary.csv`。
  下一步：后续把 suite runner 接到最终 dashboard 构建命令，形成实验到展示的一键链路。

- 实验可追溯（seed/epoch/commit/config）：✅（基础闭环）
  现状：matrix、suite、holdout multiseed 和 official runner 均输出 manifest，记录 config hash、输入/输出 hash、git、Python 环境与运行配置。
  下一步：把同一套追溯信息继续汇总到最终 dashboard/export 的说明层。

### E. 可视化与交互层

- 四联视图（原图/热力/轨迹/叠加）：✅  
  现状：已实现，且已支持单视图模式切换。  

- 方法与 case 分析：✅  
  现状：leaderboard、types、cases、method status 已展示。  

- 时间窗与阈值联动：🟡  
  现状：有热力时间窗控件，未完全驱动解释面板。  
  下一步：统一驱动 frame_event_scores。

- 事件可解释说明：🟡  
  现状：有解释面板，但事件分量链仍不完整。  
  下一步：增加 per-frame 原因强提示。

- 导出能力（json/csv/png）：☐  
  现状：尚未形成完整导出流程。  
  下一步：增设一键导出分析报告包。

### F. 部署与交付层

- 一键构建与页面发布：🟡  
  现状：可构建并已发布。  
  下一步：标准化 CLI，固定版本化发布路径。

- 中文文档与复现实验说明：🟡  
  现状：README 已较完整，但系统层面清单与流程未集中。  
  下一步：补充系统设计与执行说明。

## 3. 当前结论

- 当前系统是一个“高可视化、低端到端治理”的状态，适合展示和对比研究结果。  
- 若目标是“完整可用系统”，优先补齐顺序是：
  1) schema + 方法注册 + run manifest  
  2) 分数分量统一 + 事件段解释打通  
  3) 一键运行与导出链路固化  
  4) 自动复现与版本发布

## 4. 下一步（4 周）

- 第1周：A、D 模块（治理层）  
- 第2周：C、E 解释链路（行为异常）  
- 第3周：部署与导出链路  
- 第4周：验收与文档统一

## 2026-05-25 更新：导出链路

- 新增 `code/system/fusiontrack/export_package.py`，支持把最终 dashboard、`assets/`、pipeline summary、pipeline manifest 打包为便携 zip。
- `code/system/run_fusiontrack.py` 新增 `--export-package` 参数，可在生成系统网页后同步产出交付包。
- 导出包内使用 `${work_root}`、`${data_root}` 等占位符替代本机绝对路径，便于答辩展示、归档和跨机器交付。
- 该更新推进了 E/F 层中的“导出能力（json/csv/png/html）”与“交付链路固化”，但完整目标仍需继续补齐自动化发布、真实标签 adapter、在线/批处理运行套件等项。

## 2026-05-25 更新：评估输入 schema 治理

- 新增 `code/anomaly_detection/benchmark/evaluation/schema.py`，统一校验 label/score 行结构。
- `evaluate_score_file()` 现在会在 alignment 和指标计算前执行 fail-fast 校验。
- label 行必须包含任务 key 与二值 `label`；score 行必须包含任务 key 与有限数值 `score`。
- 支持 `sample_id` 与 `sample_id + window_id` 等任务级 key 字段，并沿用 `require_unique_keys` 策略检查重复 key。
- 该更新推进了 A/D 层中的 `labels/scores schema`、`strict key 管理` 和 `缺失字段一致性约束`，但完整系统仍需继续补齐 run manifest、真实标签 adapter、事件段解释链路和一键运行套件。

## 2026-05-25 更新：run manifest 可追溯性

- `run_benchmark_matrix.py` 输出的 `manifest.json` 升级为 `manifest_schema_version = 2`。
- 新增 `generated_at_utc`、`config_sha256`、`git`、`environment` 和 `inputs` 字段。
- 每个 run 现在记录 `experiment_config`、`score_sha256` 与 `metrics_sha256`，便于追踪方法参数、分数文件和指标文件来源。
- 该更新推进了 D/F 层中的 `seed/commit/config` 可追溯和交付归档能力；后续还需要把同类 manifest 约束扩展到 holdout multiseed、官方 baseline runner 和最终 dashboard 导出包。

## 2026-05-25 更新：holdout multiseed 聚合 manifest

- `run_fusiontrack_holdout_multiseed.py` 输出的 `manifest.json` 也升级为 `manifest_schema_version = 2`。
- 新增协议参数快照、`all_runs.csv` / `aggregate.csv` / `best_by_metric.json` 的 SHA-256、git 元数据和 Python 环境信息。
- 该更新进一步补齐最终 holdout 结果的可追溯链路；下一步应把官方 baseline runner、最终 dashboard 导出包和事件解释输出纳入同一套追溯规范。

## 2026-05-25 更新：official baseline runner manifest

- `run_recent_official_fusiontrack.py` 输出的 `run_manifest.json` 升级为 `manifest_schema_version = 2`。
- 新增官方 runner 的协议参数、超参数、git 元数据、环境信息，以及 score/convergence 文件 SHA-256。
- 该更新补齐了官方 baseline 适配运行的基础可追溯字段；后续仍需把真实官方仓库 commit/license、运行环境包版本和最终 dashboard 展示链路继续固化。

## 2026-05-25 更新：dataset manifest 数据版本治理

- 新增 `code/system/fusiontrack/dataset_manifest.py`。
- 系统入口现在会为 `smoke`、`experiment_report` 和 `final_results_dashboard` 生成 `dataset_manifest_<split>.json`。
- manifest 记录 VT-Tiny-MOT 数据根目录状态、annotation 文件 SHA-256、annotation/image/video/category 计数、图像目录文件数和 `dataset_fingerprint`。
- pipeline summary、pipeline manifest 和导出 zip 都会携带 dataset manifest 信息，便于确认实验结果来自哪一版数据结构。
- 若只是离线渲染已有结果，缺失数据根目录会记录为 `missing_data_root`，不会阻断 dashboard 构建；真正重新抽取轨迹时仍由抽取脚本 fail-fast。

## 2026-05-25 更新：synthetic protocol manifest v2

- `prepare_anomaly_data.py` 的 `--manifest-json` 输出升级为 `manifest_schema_version = 2`。
- 新 manifest 记录合成异常任务层级、key 字段、异常比例、seed、异常类型全集/子集、输入/输出/标签文件 SHA-256、label 分布和 replay argv。
- 新增 `--dataset-manifest-json` 参数，可把 dataset manifest 的 `dataset_fingerprint` 和 manifest 文件 SHA-256 写入异常注入 manifest。
- `prepare_vt_tiny_mot_protocol.py` 与 `prepare_vt_tiny_mot_holdout_protocol.py` 现在会自动生成 `dataset_manifest.json`，并强制传给 individual/group 注入 manifest。
- 该更新推进了 A 层中的“合成异常协议治理”；后续仍需把协议 manifest 汇总到最终 dashboard/export 的说明层。

## 2026-05-25 更新：真实标签并行接口

- 新增 `code/anomaly_detection/benchmark/dataset_adapters/real_labels.py`，用于把外部真实异常标注统一转为 FusionTrack label schema。
- 新增 `code/anomaly_detection/benchmark/runners/prepare_real_labels.py`，支持 CSV/JSONL 输入，输出可直接被 `run_evaluation.py` 使用的 label JSONL。
- individual 标签使用 `sample_id`；如果源文件只有 `sequence + track_id`，会自动构造 `sample_id`。
- group 标签使用 `sample_id + window_id`，缺失 `window_id` 会 fail-fast，避免把群体窗口标签退化成轨迹级标签。
- 该接口不会改变当前 synthetic 实验，只提供将来接入人工/真实异常标注时的并行入口。

## 2026-05-25 更新：run suite 批量评测入口

- 新增 `code/anomaly_detection/benchmark/runners/run_suite.py`。
- suite JSON 可以列出多个 matrix config，例如 individual 与 group 两个任务；脚本会逐个调用 `run_benchmark_matrix.py`。
- 每个 matrix 保留自己的 `manifest.json`、`summary.csv`、scores 和 metrics。
- suite 层新增 `suite_manifest.json` 和 `aggregate_summary.csv`，用于统一记录多任务评测输出路径、hash、运行数量、git 状态和 Python 环境。
- 该更新推进了 D 层中的“多任务多方法批量评测”和“实验可追溯”；后续还需要把 suite 输出直接接入最终 dashboard/export。
