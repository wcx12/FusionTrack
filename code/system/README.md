# FusionTrack 系统可视化网页说明

`code/system` 负责汇总实验结果并生成 FusionTrack 最终展示网页。当前网页是一个静态 dashboard，可以本地打开，也可以把生成后的 `index.html` 和 `assets/` 一起部署到 GitHub Pages。

## 页面定位

最终网页不是单一算法 demo，而是论文系统的展示层，主要承担这些工作：

- 展示 `Individual`、`Group`、`Registration` 三类任务的最终结果。
- 汇总方法指标、异常协议、典型案例、算法接入状态、数据流审计和动态可视化。
- 区分“视频轨迹异常检测”和“点云配准诊断”两类数据源，避免把配准任务误认为缺少原始视频。
- 支持中文 / English 切换，便于答辩、系统演示和公开静态展示。

## 生成入口

核心生成命令由 `code/system/run_fusiontrack.py` 提供。网页相关逻辑主要在：

- `code/system/fusiontrack/final_dashboard.py`：最终 dashboard HTML、前端交互和播放数据打包。
- `code/system/fusiontrack/visualization.py`：轨迹、背景帧和基础可视化工具。
- `code/system/fusiontrack/registration_adapter.py`：配准实验结果接入 dashboard。
- `code/system/fusiontrack/final_results.py`：最终结果表、排行榜、case 和方法分类汇总。
- `code/system/fusiontrack/explanation_schema.py`：后端解释 schema，统一输出主导原因、证据来源、事件阈值策略和分数分量。
- `code/system/fusiontrack/dataset_manifest.py`：生成数据集结构、annotation hash、图像目录计数和数据集指纹。
- `code/system/fusiontrack/method_registry.py`：读取和校验中心方法注册表。
- `code/anomaly_detection/benchmark/configs/method_registry.json`：中心方法注册表，是方法归属、角色、方法族、学习类型、来源类型和接入状态的权威来源。

推荐用配置文件固化最终 dashboard 的长命令：

```bash
python code/system/run_fusiontrack.py --run-config code/system/configs/final_dashboard.local.example.json
```

`--run-config` 接收 JSON 对象，字段名与 CLI 参数的下划线形式一致，例如 `final_results_root`、`individual_label_file`、`score_search_roots`、`protocol_manifests`、`top_sequences`。配置中的相对路径会按 `base_dir` 解析；示例配置使用 `base_dir: "../../.."`，因此所有路径都是仓库根目录相对路径，不需要把本机绝对路径写入配置。临时覆盖某个参数时，可以在配置文件后继续追加 CLI 参数，例如：

```bash
python code/system/run_fusiontrack.py \
  --run-config code/system/configs/final_dashboard.local.example.json \
  --data-root data/VT-Tiny-MOT \
  --top-sequences 3
```

常见生成产物：

- `server_artifacts/remote_result/report/index.html`
- `server_artifacts/remote_result/report/assets/final_dashboard_data.json`
- `server_artifacts/remote_result/report/assets/final_playback_data.json`
- `server_artifacts/remote_result/report/assets/background_*.jpg`
- `runs/<work_root>/dataset_manifest_<split>.json`

注意：`server_artifacts/` 默认被 `.gitignore` 忽略，不会提交到主分支。公开部署时必须把 `index.html`、`assets/final_*.json` 和 `assets/background_*.jpg` 作为同一套静态资源一起发布。

## 页面模块

### 顶部控制区

顶部提供四类主要控件：

- 语言：中文 / English。
- 任务：`Individual` / `Group` / `Registration`。
- 方法：当前任务下接入的算法或基线。
- 序列：当前任务下可播放或可诊断的序列。

切换任务时，页面会同步切换方法列表、序列列表、指标卡片、播放视图和分析表格。

### 指标卡片

`Individual` 和 `Group` 显示：

- 方法数。
- 当前任务总标签数。
- 当前任务总异常数。
- 当前选中方法的 AUROC。

`Registration` 显示：

- 配准方法数。
- 配准样本数。
- 失败或跳过样本数。
- 当前选中方法的成功率。

`Registration` 当前不是异常标签评测任务，而是几何配准诊断任务，因此主指标不是 AUROC。

### 异常协议概览

VT-Tiny-MOT 原始数据没有异常标签。当前 `Individual` 和 `Group` 的异常标签来自规则化 synthetic anomaly injection：

- `Individual`：路线偏移、速度突变、停止或减速、跳变、形状扭曲、RGB/thermal 模态偏移。
- `Group`：离群、逆向运动、邻居替换、群体数量变化、离散程度变化、分裂或合并。

`Registration` 单独展示点云配准质量，包括旋转误差、平移误差、Chamfer、耗时和成功率。它不使用 VT-Tiny-MOT 原始视频帧。

## 动态可视化

### Individual / Group：四画面对比

视频型任务默认使用四画面对比：

- 原视频：只显示原始 RGB 背景帧。
- 热力图：在原始背景上叠加异常热力。
- 轨迹：在原始背景上叠加轨迹线。
- 热力 + 轨迹：同时叠加热力和轨迹，作为默认综合观察方式。

这些画面不是 `<video>` 标签播放 MP4，而是前端用 `canvas` 把背景帧、轨迹点和热力图逐帧绘制出来。只要当前序列是 `DJI_****` 这类真实 VT-Tiny-MOT 视频序列，并且 `data_root` 可解析 `rgb.file`，页面会复制最多 72 张原始背景帧到 `assets/` 目录。

支持的交互包括：

- 播放 / 暂停。
- 键盘快捷控制：空格播放/暂停，左右方向键逐帧移动，Home/End 跳到首尾帧，数字键切换四画面或单画面图层，加减键调整播放速度。
- 帧滑块。
- 四画面对比 / 单画面模式。
- 单画面图层选择。
- 热力透明度。
- 时间窗口。
- 播放速度。
- 点击轨迹选择当前解释对象。

### Registration：点云配准动态视图

`Registration` 不使用原视频四宫格。切换到 `Registration` 后，页面显示独立的点云配准诊断视图：

- 源点云 `source`。
- 参考点云 `reference`。
- 估计对齐结果 `aligned`。
- 当前点云对、方法、旋转误差、平移误差、Chamfer、耗时、成功/失败状态和风险分数。

这个视图本质是点云诊断 canvas，不依赖原始视频背景。如果 score row 暂时没有真实 `registration_points`，页面会使用轻量占位点云保证展示层不断裂；后续接入真实 MPS-GAF 或其他学习式配准输出后，应优先使用真实 `source/reference/aligned` 点云替换占位数据。

## 背景资源与媒体类型

每个可播放序列都会在 `final_playback_data.json` 中带有 `media` 字段：

- `media.kind == "original_video_background"`：序列有原始视频背景帧，适合四画面对比。
- `media.kind == "registration_point_cloud"`：序列是点云配准诊断样本，没有 VT-Tiny-MOT 原始视频背景。
- `media.kind == "track_only_missing_background"`：序列有轨迹点，但没有找到可用 RGB 背景帧，需要检查 `data_root` 或 fused trajectory 中的 `rgb.file`。

相关字段：

- `media.has_original_background`：是否有原始背景。
- `media.background_frame_count`：已复制到网页资产中的背景帧数量。
- `media.explanation_key`：前端提示文案 key。
- `modality_audit.background_status`：数据流审计中的背景状态。
- `background_frames[].src`：当前帧附近优先加载的抽样原始背景帧。
- `background_frames[].fallback_src`：抽样帧缺失或静态部署漏传时的序列首帧回退背景。

因此，如果网页里某些“视频”没有背景，需要先看当前任务和序列：

- `DJI_****`：真实视频序列，通常应该有原始 RGB 背景。
- `batch_****` 或配准 batch：点云配准实验样本，本来就没有原始视频背景。

如果 `DJI_****` 序列仍然提示无背景，才需要检查数据集路径、资源部署和 `rgb.file` 是否能解析到真实图片。

### 背景加载回退机制

最终网页不会直接播放完整 MP4，而是逐帧绘制 `assets/background_*.jpg` 或 `assets/background_*.png`。为了避免静态部署时某个抽样背景帧漏传导致整块 canvas 变成“加载中”，当前生成器会给每个抽样帧写入 `fallback_src`：

- 优先加载当前帧对应的 `background_<sequence>_<frame>.jpg`。
- 如果该帧加载失败，自动回退到 `background_<sequence>.jpg`。
- 如果主背景也加载失败，页面显示“背景帧加载失败，请检查静态网页 assets 是否已同步发布”。

公开部署时必须保持 `index.html` 和 `assets/` 的相对目录结构不变。只提交 `index.html` 或只提交 `final_playback_data.json` 都不够，背景帧、最终数据 JSON 和页面文件必须来自同一次生成结果。

## 事件解释链路

最终 dashboard 会读取 score row 中的事件证据字段：

- `event_score`：当前轨迹或对象的事件级最高风险分数。
- `event_segments`：算法侧已经合并好的事件段，包含 `frame_start`、`frame_end`、`score` 和主导原因。
- `frame_event_scores`：逐帧事件证据序列，包含帧号、逐帧分数、主导原因和分量分数。
- `explanation_schema`：后端生成的稳定解释结构，包含 `top_reason`、`evidence_source`、`policy`、`peak_event` 和 `score_components`。

事件段合并现在由 `code/system/fusiontrack/event_segments.py` 统一处理：

- `normalize_frame_event_scores()` 会过滤非法帧号、非有限分数，并统一 `frame`、`score`、`dominant_reason`、`component_scores` 和 `source`。
- `event_segments_from_frame_scores()` 会按阈值筛选正事件帧，允许配置小间隔合并，保留峰值分数、主导原因、持续帧数和分量最大值。

如果方法已经输出 `event_segments`，页面会直接使用该字段绘制预测事件段。如果方法只输出 `frame_event_scores`，`score_fusion.py` 和 `final_dashboard.py` 会先在后端合并生成 `event_segments`；前端的 `eventSegmentsFromFrameScores()` 只保留为兼容旧数据的兜底逻辑。这样可以让热力时间窗口、事件时间线、右侧解释面板和导出的 score JSONL 围绕同一份逐帧证据工作。

`explanation_schema.py` 会在后端统一确定解释主因：优先使用峰值事件段的 `dominant_reason`，没有事件证据时回退到分数分量中贡献最大的组件。该 schema 同时记录事件阈值、最大合并间隔、最小事件长度等策略字段，因此前端解释面板不再需要自行猜测 top reason，只负责把后端解释结构渲染出来。

当前轻量 individual 检测器和群体图打分方法都会输出 `frame_event_scores`、`event_segments` 和 `explanation_schema`。其中 individual 分支会把 speed spike、turn irregularity、low confidence、modal offset 等分量转成逐帧事件证据；后续更强的 route/speed/shape 学习模型也应继续沿用相同字段，保证前端解释面板和事件时间线不需要改接口。

融合分数 row 也会保留这些事件字段，并在 `component_scores` 中补充：

- `S_ind`：归一化后的 individual 分支分数。
- `S_grp`：归一化后的 group 分支分数。
- `S_event`：来自 individual/group 事件证据的最高事件分数。
- `S_fused`：最终融合分数。
- `explanation_schema`：融合 row 的后端解释结果，供最终 dashboard 的分数分解条、事件时间线和解释面板共用。

最终 dashboard 会优先读取这些显式 `S_*` 字段；来源判断同时支持顶层 `used_sources` 和 `metadata.used_sources`，因此 `score_fusion.py` 导出的融合 JSONL 可以直接驱动网页分数分解条。

这样最终网页的分数分解条、事件时间线和导出的 score JSONL 使用的是同一条解释链。

## 实验分析区

页面下半部分包含：

- 方法排名：展示当前任务下各方法指标。
- 异常类型分析：展示不同异常类型的命中情况。
- 典型案例：展示 true positive、false positive、false negative 或高风险样本。
- 算法接入：解释方法来源、角色、学习类型和覆盖情况。
- 数据流审计：检查每个任务和序列的标签、分数、轨迹点、背景帧、RGB/thermal 覆盖率。

这些模块用于回答“当前系统接入了哪些结果、哪些数据完整、哪些方法只是 proxy 或 baseline”的问题。

审计版页面还提供轻量导出功能：

- 当前视图 JSON：导出当前任务、方法、序列、帧号、选中轨迹、分数解释和指标上下文。
- 方法排行榜 CSV：导出当前任务下的方法排名表。
- 当前序列 JSON：导出当前序列的轨迹、标签、分数、背景帧和数据审计信息。
- 当前画面 PNG：导出当前播放画面；四画面对比模式会生成原视频、热力、轨迹、热力+轨迹的 2x2 拼图，单画面模式导出当前图层，Registration 模块导出点云配准画面。
- 完整报告包 ZIP：当构建命令传入 `--export-package` 时，页面右上角会出现“下载完整报告包”入口；该 ZIP 会包含 dashboard、脱敏后的 summary/manifest 以及被引用的实验产物，便于答辩离线演示或归档。

PNG 导出依赖浏览器 Canvas，因此公开部署时仍要保证 `assets/` 背景帧与页面同源发布，避免浏览器把 canvas 标记为不可导出。

## 方法注册表治理

最终结果看板会同时读取最终实验 CSV、score JSONL 和中心方法注册表。为避免旧的 categorized CSV 把方法身份写错，当前规则是：

- `method_registry.json` 是方法治理字段的唯一权威来源。
- 权威字段包括 `owner`、`role`、`method_family`、`learning_type`、`source_type`、`status`、`aliases` 和 `registry_status`。
- `final_*_categorized.csv` 只能补充历史指标或非治理字段，不能覆盖中心注册表中的方法身份。
- 未登记方法会被标记为 `registry_status == "unregistered"`，并显示为 `owner == "unregistered"`。
- 已登记方法会被标记为 `registry_status == "registered"`。

新增或更名方法时，应先更新 `method_registry.json`，再生成最终结果和网页。注册表支持 `aliases`，用于把历史方法名或实验产物中的临时方法名映射到正式方法名。

可以用下面的方式检查注册表：

```bash
cd code/system
python -c "from fusiontrack.method_registry import validate_method_registry; print(validate_method_registry())"
```

校验会检查必填字段、重复方法、重复 alias、alias 是否撞到同任务下的其他方法名。真实注册表应保持 `status == "ok"`，这样排行榜、算法接入表和 README 中的方法分类才不会互相矛盾。

## 数据集 Manifest 治理

系统入口会为每次运行生成 `dataset_manifest_<split>.json`。最终 dashboard 使用 `dataset_manifest_all.json`，普通实验报告使用对应 split 的 manifest。

manifest 记录：

- 数据集名称和 schema version。
- `annotations` 或 `annotations_tc` 目录是否存在。
- RGB / thermal annotation 文件路径、大小、SHA-256、image / annotation / video / category 数量。
- `train2017`、`test2017`、`val2017` 图像目录是否存在、图片文件数和后缀统计。
- `dataset_fingerprint`，用于判断两次实验是否基于同一套 annotation 与图像目录结构。

如果数据根目录不存在，manifest 不会让页面构建直接失败，而是记录 `status == "missing_data_root"`；这样可以支持只渲染已有结果的离线场景。真正需要从数据集重新抽取轨迹时，抽取脚本仍会按缺失文件直接报错。

导出包会自动包含 dataset manifest，并把本机绝对路径脱敏为 `${work_root}`、`${data_root}` 或 `${external}` 占位符。

## Strict Key Policy 治理

最终结果看板会在每个任务的公共 payload 中写入 `key_policy`，用于说明标签和分数行如何对齐：

- `Individual`：严格键为 `sample_id`，表示单条轨迹级样本。
- `Group`：严格键为 `sample_id + window_id`，表示群体窗口级样本；为了兼容历史 score row，页面也会标注 `sample_id` 作为旧结果回退字段。
- `Registration`：严格键为 `sample_id`，表示一个点云配准 pair。

这些策略会同时出现在：

- `assets/final_dashboard_data.json` 的 `tasks.<task>.key_policy`。
- 审计版页面的数据流审计表。
- 当前视图 JSON 导出的 `key_policy`。
- 当前序列 JSON 导出的 `task_key_policies`。

这样答辩或复现实验时可以直接说明：指标评估、标签覆盖、分数覆盖和前端展示并不是随意按文件行号对齐，而是有明确的任务级 key policy。后续如果新增真实标注或新的 group window preset，应优先保持该 key policy 不变；如必须改变，应同步更新评估配置、score row、dashboard payload 和文档。

## 运行来源审计

最终 dashboard 的“数据流审计”页顶部会展示一组运行来源信息，用来回答“当前网页到底由哪一版数据、哪一批标签、哪一批分数和哪些构建参数生成”：

- 数据集名称、数据集状态、split 列表和 `dataset_fingerprint`。
- `dataset_manifest_<split>.json` 的公开路径提示。
- 最终结果目录、individual/group 标签文件、score 搜索目录数量、融合轨迹文件和 registration manifest。
- 如果构建命令传入 `--protocol-manifest`，会展示 synthetic anomaly injection 协议 manifest、任务、标签分布、异常类型和 dataset fingerprint。
- 如果构建命令传入 `--suite-manifest`，还会展示评测套件名称、suite manifest、aggregate summary、矩阵数量和总 run 数。
- 如果构建命令传入 `--holdout-manifest`，还会展示 holdout 多 seed 结果、最佳方法和来源文件。
- 构建参数，例如 `top_sequences`、`top_k` 和 `case_limit`。

该模块只发布脱敏后的路径提示：如果输入是本机绝对路径，网页数据里只保留文件名或目录名；如果输入本来就是相对路径，则保留相对路径。这样公开部署到 GitHub Pages 时不会泄露本机目录，同时仍能说明结果来源。

当需要把 `run_suite.py` 的批量评测结果纳入最终交付链路时，在最终网页生成命令中追加：

```bash
python code/system/run_fusiontrack.py \
  --final-results-root <final_results_root> \
  --individual-label-file <individual_labels.jsonl> \
  --group-label-file <group_labels.jsonl> \
  --fused-jsonl <merged_fused.jsonl> \
  --fused-pipeline-manifest <fused_track_pipeline_manifest.json> \
  --protocol-manifest <protocol_dir>/protocol_manifest.json \
  --suite-manifest <suite_output_dir>/suite_manifest.json \
  --holdout-manifest <holdout_output_dir>/manifest.json \
  --export-package <output.zip>
```

这样 `pipeline_summary_final_dashboard.json`、`pipeline_manifest_final_results_dashboard_all.json`、网页的 provenance 数据和导出 zip 都会保留 fused track pipeline、protocol、suite 与 holdout 来源；导出包会把 fused track pipeline manifest、protocol manifest、suite manifest 及其引用的 aggregate summary、matrix summary/manifest、holdout aggregate/all-runs/best-by-metric 一起归档。

如果传入 `--fused-pipeline-manifest`，最终 dashboard 会在数据流审计中新增轨迹融合 pipeline 面板，展示：

- 输入 observations 文件及 SHA-256。
- raw / kept / fused / filtered trajectory 计数。
- group window 数量。
- fused ratio、paired ratio 等模态覆盖。
- `TrackQualityConfig` 过滤阈值。
- `drop_reason_counts` 过滤原因统计。
- fused trajectories、group windows、summary 等输出 artifact。

## 验证建议

修改网页后至少运行：

```bash
python -m py_compile code/system/fusiontrack/final_dashboard.py code/system/fusiontrack/explanation_schema.py
python -m py_compile code/system/fusiontrack/final_results.py code/system/fusiontrack/method_registry.py code/system/fusiontrack/dataset_manifest.py
python -m py_compile code/system/tools/build_sample_dashboard.py code/system/tools/publish_dashboard_pages.py code/system/tools/build_dashboard_release.py
python -c "import collections, collections.abc; collections.Callable = collections.abc.Callable; import pytest, sys; sys.exit(pytest.main(['code/system/tests/test_explanation_schema.py', 'code/system/tests/test_dataset_manifest.py', 'code/system/tests/test_method_registry.py', 'code/system/tests/test_final_results.py', 'code/system/tests/test_pipeline.py', '-q']))"
```

仓库还提供了 `.github/workflows/system-ci.yml`，当 `code/system/**`、中心方法注册表或该 workflow 自身变更时，会在 GitHub Actions 中自动执行：

- Python 编译检查。
- `method_registry.json` 注册表校验。
- `code/system/tests` 系统测试。
- 生成一份小样例 dashboard，并以 `sample-dashboard` artifact 上传，便于在没有真实数据集和服务器产物的 CI 环境里检查静态页面是否能完整构建。

本地也可以手动生成这份小样例：

```bash
python code/system/tools/build_sample_dashboard.py --output-dir runs/sample_dashboard_ci
```

## GitHub Pages 发布与版本归档

如果希望把“构建 dashboard、生成导出包、发布 GitHub Pages、写交付清单”作为一次交付动作，可以使用一键交付命令：

```bash
python code/system/tools/build_dashboard_release.py \
  --run-config code/system/configs/final_dashboard.local.example.json \
  --export-package runs/releases/fusiontrack_20260528_final_dashboard.zip \
  --pages-dir ../FusionTrack-gh-pages \
  --run-id 20260528_final_dashboard
```

该命令会先调用 `run_fusiontrack.py --run-config ...`，再根据 pipeline summary 中的 dashboard 输出目录调用 Pages 发布工具。运行完成后，会在 `work_root` 下写入 `dashboard_release_<run_id>.json`，记录本次交付的构建命令、pipeline summary、manifest、dashboard 目录、导出包和 Pages 发布结果。该交付清单使用相对路径或 `${external}` 占位符，不写入本机绝对路径。

当命令传入 `--export-package` 时，生成的 ZIP 会同步复制到 dashboard 的 `assets/` 目录，并随 GitHub Pages 根页面和 `history/<run_id>/` 一起发布。网页中的下载入口只使用 `assets/<zip_name>` 这样的相对链接，不写入本机路径或服务器绝对路径。

生成最终 dashboard 后，可以用发布工具把静态网页同步到 GitHub Pages 工作树。命令建议只写相对路径，避免把本机绝对路径带入脚本、服务器命令或公开清单：

```bash
python code/system/tools/publish_dashboard_pages.py \
  --source-dir runs/final_results_dashboard/final_dashboard \
  --pages-dir ../FusionTrack-gh-pages \
  --run-id 20260528_final_dashboard
```

该工具会执行三件事：

- 更新 Pages 根目录的 `index.html` 与完整 `assets/`，保证公开首页访问的是最新 dashboard。
- 同步保留 `CNAME` 等根目录已有文件，只替换 dashboard 相关静态资源。
- 将同一版页面归档到 `history/<run_id>/`，便于答辩或论文归档时回看不同实验批次。

发布后会在 Pages 根目录写入 `publish_manifest.json`。该清单只记录相对公开路径、`run_id`、资产数量和发布时间，不写入本机绝对路径。常见检查点如下：

- `index.html` 存在。
- `assets/final_dashboard_data.json` 和 `assets/final_playback_data.json` 存在。
- `assets/background_*.jpg` 等背景资源随页面一起发布。
- `history/<run_id>/index.html` 与 `history/<run_id>/assets/` 存在。
- `publish_manifest.json` 中没有 `D:/...`、`C:\...` 或服务器绝对路径。

生成网页后建议用浏览器或 Playwright 截图验证：

- `Individual` 默认四画面对比有原视频背景。
- `Group` 可以正常显示原视频、热力和轨迹。
- `Registration` 不显示原视频四宫格，而显示点云配准动态视图。
- 中英文切换不破坏布局。
- 小屏和桌面视口没有明显横向溢出和文本重叠。

## 当前限制

- `Individual` 和 `Group` 的异常标签是规则注入标签，不是数据集原生标注。
- `Registration` 当前主要是配准诊断模块，不是异常检测 AUROC 任务。
- 部分配准点云预览仍可能是轻量展示数据，后续应接入真实模型输出的 `source/reference/aligned` 点云。
- 当前网页是静态页面，没有后端在线推理接口；重新跑实验后需要重新生成并部署页面。
