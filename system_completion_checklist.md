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

- labels/scores schema 与校验：🟡  
  现状：有 `jsonl/csv` 读写，但缺统一 schema 层。  
  下一步：新增 schema 定义 + 校验失败 fail-fast。

- strict key 管理：🟡  
  现状：protocol 中定义了 `sample_id` / `sample_id+window_id` 口径。  
  下一步：在评估入口集中强制统一。

- 合成异常协议治理：✅（主协议闭环）
  现状：异常注入脚本已输出 manifest v2，validation/holdout 协议生成器会自动生成 `dataset_manifest.json` 并传入注入 manifest，记录参数、文件 hash、label 分布、重放命令和 dataset fingerprint。
  下一步：把协议 manifest 汇总到最终 dashboard/export 的解释层。

- 真实标签并行接口：☐  
  现状：主要以 synthetic 为主。  
  下一步：增加真实标注 adapter 并统一入口字段。

### B. 融合与轨迹构建层

- 多模态标准化：🟡  
  现状：章节有完整思路，工程里未统一成标准服务。  
  下一步：抽标准化模块，固定输出字段。

- 跨模态关联与跨帧关联：🟡  
  现状：已有实现思路与展示链路，缺乏统一 pipeline 封装。  
  下一步：形成可复用 `fused track pipeline`。

- 噪声抑制与目标持久化策略：🟡  
  现状：可视化有描述。  
  下一步：参数化配置并与实验日志绑定。

- 融合轨迹可追溯输出：🟡  
  现状：可用于可视化。  
  下一步：统一输出目录与版本标识。

### C. 行为异常检测层

- Individual 分支（route/speed/shape）：☐  
  现状：论文方案完整，工程聚合层已在展示侧，算法层未完全闭环到统一输出。  
  下一步：补充分量分数统一打通到 dashboard payload。

- Group 分支（群体结构与事件）：🟡  
  现状：有群体关系与相关可视化。  
  下一步：事件分量与 score 详细来源打通。

- 个体-群体融合：☐  
  现状：存在 fused-score 思路，但前端并非强制读取融合分量链。  
  下一步：统一 `S_ind / S_grp / S_event` 展示与导出。

- 事件段生成与平滑：☐  
  现状：未在主展示面板形成完整事件段语义。  
  下一步：实现 frame 级 score 序列与事件段合并输出。

### D. 评测与治理层

- 方法注册表：🟡  
  现状：页面展示 owner/role/family，但未集中于单一 registry。  
  下一步：建立 `method_registry` 作为唯一来源。

- 指标聚合：✅（部分）  
  现状：AUROC/AUPRC/F1/P@K/R@K 已支持。  
  下一步：补缺失字段告警与一致性约束。

- 多任务多方法批量评测：🟡  
  现状：有矩阵式流程。  
  下一步：封装 `run_suite` 一键运行脚本。

- 实验可追溯（seed/epoch/commit/config）：🟡  
  现状：散落于日志与 README。  
  下一步：引入 run manifest 并入结果目录。

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
