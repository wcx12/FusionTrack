# FusionTrack 系统可视化网页说明

本目录负责生成 FusionTrack 的最终系统展示网页。当前网页是一个静态 dashboard，生成后可直接部署到 GitHub Pages，也可以用浏览器打开本地 `index.html` 查看。

## 页面定位

最终网页不是单一算法 demo，而是论文系统展示层，主要承担四件事：

- 展示 Individual、Group、Registration 三类任务的最终结果。
- 把方法指标、异常协议、典型案例、数据流审计和可视化播放放在同一个页面里。
- 区分“视频轨迹异常检测”和“点云配准诊断”两类不同数据源，避免把配准任务误看成缺少原始视频。
- 支持中英文切换，便于论文答辩、系统演示和 GitHub Pages 公开展示。

## 页面入口

生成入口是：

```bash
python code/system/run_fusiontrack.py
```

核心 HTML 生成逻辑在：

- `code/system/fusiontrack/final_dashboard.py`
- `code/system/fusiontrack/visualization.py`
- `code/system/fusiontrack/registration_adapter.py`

生成结果通常位于：

- `server_artifacts/remote_result/report/index.html`
- `server_artifacts/remote_result/report/assets/final_dashboard_data.json`
- `server_artifacts/remote_result/report/assets/final_playback_data.json`

其中 `server_artifacts/` 是本地/服务器生成物目录，默认不提交到主分支；部署到 GitHub Pages 时会把静态网页复制到 `gh-pages` 分支。

## 当前页面模块

### 1. 顶部控制区

顶部提供四个主要控件：

- 语言：中文 / English。
- 任务：Individual / Group / Registration。
- 方法：当前任务下的可选算法。
- 序列：当前任务下可播放或可诊断的序列。

切换任务时，页面会同步切换方法列表、序列列表、指标卡片、播放视图和分析表格。

### 2. 指标卡片

Individual 和 Group 任务显示：

- 方法数。
- 当前任务总标签数。
- 当前任务总异常数。
- 当前选中方法 AUROC。

Registration 任务显示：

- 配准方法数。
- 配准样本数。
- 失败或跳过样本数。
- 当前选中方法成功率。

注意：Registration 当前不是异常标签任务，而是几何配准诊断任务，因此不使用 AUROC 作为主卡片指标。

### 3. 异常协议概览

页面说明当前异常标签来自规则化 synthetic anomaly injection。VT-Tiny-MOT 原始数据没有异常标签，系统通过规则注入得到 Individual 和 Group 的评测标签。

Individual 主要覆盖单目标轨迹异常，例如：

- 路线偏移。
- 速度突变。
- 停止或减速。
- 跳变。
- 形状扭曲。
- RGB/thermal 模态偏移。

Group 主要覆盖群体关系异常，例如：

- 离群。
- 逆向运动。
- 邻居替换。
- 群体数量变化。
- 离散程度变化。
- 分裂或合并。

Registration 单独展示点云配准质量，包括旋转误差、平移误差、Chamfer、耗时和成功率。它不使用 VT-Tiny-MOT 原始视频帧。

## 动态可视化区

### Individual / Group：四画面对比

视频型任务默认使用四画面对比：

- 原视频：只显示原始 RGB 背景帧。
- 热力图：在原始背景上叠加异常热力。
- 轨迹：在原始背景上叠加轨迹线。
- 热力 + 轨迹：同时叠加热力和轨迹，作为默认综合观察方式。

这些背景帧来自 `final_playback_data.json` 中每个 VT-Tiny-MOT 序列的 `background_frames` 字段。只要序列是 `DJI_****` 这类真实视频序列，并且数据集路径可用，页面会复制最多 72 张背景帧到网页 assets 目录。

可交互控件包括：

- 播放 / 暂停。
- 帧滑块。
- 四画面对比 / 单画面模式。
- 单画面图层选择。
- 热力透明度。
- 时间窗口。
- 播放速度。
- 点击轨迹选择当前解释对象。

### Registration：点云配准动态视图

Registration 不再使用原视频四宫格。切换到 Registration 后，页面显示独立的点云配准视图：

- 源点云 source。
- 参考点云 reference。
- 估计对齐结果 aligned。
- 当前点云对、方法、旋转误差、平移误差、Chamfer、耗时、成功/失败状态和风险分数。

播放按钮在 Registration 下用于让点云投影随帧号产生轻量旋转，帮助观察 source/reference/aligned 的相对位置。该视图本质上是点云诊断 canvas，不依赖原始视频背景。

如果某些 score row 暂时没有真实 `registration_points`，页面会使用轨迹点生成轻量占位点云，保证展示层不断裂。后续接入真实 MPS-GAF 或其他学习式配准输出后，应优先使用真实 source/reference/aligned 点云替换占位数据。

## 为什么有些序列没有原始背景

这是设计上的任务差异，不是资源上传失败。

- `DJI_****`：VT-Tiny-MOT 视频序列，通常有原始 RGB 背景帧。
- `batch_****` 或配准 batch：点云配准实验样本，没有 VT-Tiny-MOT 原视频背景。

因此：

- Individual / Group 应显示原视频背景。
- Registration 应显示点云配准视图。
- 如果某个真实视频序列提示无背景，才需要检查 `data_root` 是否指向 VT-Tiny-MOT 数据集，以及 fused trajectory 中 `rgb.file` 是否能解析到真实图片。

## 实验分析区

页面下半部分包含：

- 方法排名：显示当前任务下各方法指标。
- 异常类型分析：显示不同异常类型的命中情况。
- 典型案例：显示 true positive、false positive、false negative 或高风险样本。
- 算法接入：解释方法来源、角色、学习类型和覆盖情况。
- 数据流审计：检查每个任务和序列的标签、分数、轨迹点、背景帧、RGB/thermal 覆盖率。

这些模块用于回答“当前系统接了哪些结果、哪些数据完整、哪些方法只是 proxy 或 baseline”的问题。

## 验证建议

修改网页后至少运行：

```bash
python -m py_compile code/system/fusiontrack/final_dashboard.py
python -c "import collections, collections.abc; collections.Callable = collections.abc.Callable; import pytest, sys; sys.exit(pytest.main(['code/system/tests', '-q']))"
```

生成网页后建议用 Playwright 截图验证：

- Individual 默认四画面对比有原视频背景。
- Group 可正常显示原视频、热力和轨迹。
- Registration 不显示原视频四宫格，而显示点云配准动态视图。
- 中英文切换不会破坏布局。
- 375px、768px、1280px 视口下没有明显横向溢出和文本重叠。

## 当前限制

- Individual 和 Group 的异常标签是规则注入标签，不是数据集原生标注。
- Registration 当前主要是配准诊断模块，不是异常检测 AUROC 任务。
- Registration 的部分点云预览仍可能是轻量展示数据，后续应接入真实模型输出的 source/reference/aligned 点云。
- 当前网页是静态页面，没有后端在线推理接口；重新跑实验后需要重新生成并部署页面。

## 导出交付包

最终 dashboard 生成后可以额外导出一个便携 zip 包，用于答辩、归档或在其它机器上离线查看核心结果：

```bash
python code/system/run_fusiontrack.py \
  --final-results-root <final_results_root> \
  --individual-label-file <individual_labels.jsonl> \
  --group-label-file <group_labels.jsonl> \
  --fused-jsonl <fused_trajectories.jsonl> \
  --export-package runs/fusiontrack_v1/exports/fusiontrack_dashboard_export.zip
```

导出包由 `fusiontrack.export_package.build_analysis_export_package` 生成，包含：

- `report/index.html`：最终可视化网页。
- `report/assets/`：网页需要的 JSON、背景帧、曲线或图片资源。
- `summary/pipeline_summary.json`：脱敏后的 pipeline summary。
- `summary/pipeline_manifest.json`：脱敏后的运行 manifest。
- `export_manifest.json`：导出包清单、文件大小、包格式和相对化来源路径。

导出包内部不会写入本机绝对路径；`work_root`、`data_root` 等路径会被替换为 `${work_root}`、`${data_root}` 这类可迁移占位符。
