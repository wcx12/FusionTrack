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
- `code/system/fusiontrack/dataset_manifest.py`：生成数据集结构、annotation hash、图像目录计数和数据集指纹。
- `code/system/fusiontrack/method_registry.py`：读取和校验中心方法注册表。
- `code/anomaly_detection/benchmark/configs/method_registry.json`：中心方法注册表，是方法归属、角色、方法族、学习类型、来源类型和接入状态的权威来源。

推荐用配置文件固化最终 dashboard 的长命令：

```bash
python code/system/run_fusiontrack.py --run-config code/system/configs/final_dashboard.local.example.json
```

`--run-config` 接收 JSON 对象，字段名与 CLI 参数的下划线形式一致，例如 `final_results_root`、`individual_label_file`、`score_search_roots`、`top_sequences`。配置中的相对路径会按 `base_dir` 解析；示例配置使用 `base_dir: "../../.."`，因此所有路径都是仓库根目录相对路径，不需要把本机绝对路径写入配置。临时覆盖某个参数时，可以在配置文件后继续追加 CLI 参数，例如：

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

事件段合并现在由 `code/system/fusiontrack/event_segments.py` 统一处理：

- `normalize_frame_event_scores()` 会过滤非法帧号、非有限分数，并统一 `frame`、`score`、`dominant_reason`、`component_scores` 和 `source`。
- `event_segments_from_frame_scores()` 会按阈值筛选正事件帧，允许配置小间隔合并，保留峰值分数、主导原因、持续帧数和分量最大值。

如果方法已经输出 `event_segments`，页面会直接使用该字段绘制预测事件段。如果方法只输出 `frame_event_scores`，`score_fusion.py` 和 `final_dashboard.py` 会先在后端合并生成 `event_segments`；前端的 `eventSegmentsFromFrameScores()` 只保留为兼容旧数据的兜底逻辑。这样可以让热力时间窗口、事件时间线、右侧解释面板和导出的 score JSONL 围绕同一份逐帧证据工作。

当前群体图打分方法已经输出 `frame_event_scores` 和 `event_segments`；后续 individual route/speed/shape 分支也应采用相同字段接入。

融合分数 row 也会保留这些事件字段，并在 `component_scores` 中补充：

- `S_ind`：归一化后的 individual 分支分数。
- `S_grp`：归一化后的 group 分支分数。
- `S_event`：来自 individual/group 事件证据的最高事件分数。
- `S_fused`：最终融合分数。

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

## 运行来源审计

最终 dashboard 的“数据流审计”页顶部会展示一组运行来源信息，用来回答“当前网页到底由哪一版数据、哪一批标签、哪一批分数和哪些构建参数生成”：

- 数据集名称、数据集状态、split 列表和 `dataset_fingerprint`。
- `dataset_manifest_<split>.json` 的公开路径提示。
- 最终结果目录、individual/group 标签文件、score 搜索目录数量、融合轨迹文件和 registration manifest。
- 如果构建命令传入 `--suite-manifest`，还会展示评测套件名称、suite manifest、aggregate summary、矩阵数量和总 run 数。
- 构建参数，例如 `top_sequences`、`top_k` 和 `case_limit`。

该模块只发布脱敏后的路径提示：如果输入是本机绝对路径，网页数据里只保留文件名或目录名；如果输入本来就是相对路径，则保留相对路径。这样公开部署到 GitHub Pages 时不会泄露本机目录，同时仍能说明结果来源。

当需要把 `run_suite.py` 的批量评测结果纳入最终交付链路时，在最终网页生成命令中追加：

```bash
python code/system/run_fusiontrack.py \
  --final-results-root <final_results_root> \
  --individual-label-file <individual_labels.jsonl> \
  --group-label-file <group_labels.jsonl> \
  --fused-jsonl <merged_fused.jsonl> \
  --suite-manifest <suite_output_dir>/suite_manifest.json \
  --export-package <output.zip>
```

这样 `pipeline_summary_final_dashboard.json`、`pipeline_manifest_final_dashboard_all.json`、网页的 provenance 数据和导出 zip 都会保留 suite 评测来源；导出包会把 suite manifest 及其引用的 aggregate summary、matrix summary/manifest 一起归档。

## 验证建议

修改网页后至少运行：

```bash
python -m py_compile code/system/fusiontrack/final_dashboard.py
python -m py_compile code/system/fusiontrack/final_results.py code/system/fusiontrack/method_registry.py code/system/fusiontrack/dataset_manifest.py
python -c "import collections, collections.abc; collections.Callable = collections.abc.Callable; import pytest, sys; sys.exit(pytest.main(['code/system/tests/test_dataset_manifest.py', 'code/system/tests/test_method_registry.py', 'code/system/tests/test_final_results.py', 'code/system/tests/test_pipeline.py', '-q']))"
```

仓库还提供了 `.github/workflows/system-ci.yml`，当 `code/system/**`、中心方法注册表或该 workflow 自身变更时，会在 GitHub Actions 中自动执行：

- Python 编译检查。
- `method_registry.json` 注册表校验。
- `code/system/tests` 系统测试。

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
