# FusionTrack 结果展示面板说明（最终版本）

## 1. 文件位置

本次可视化面板的主入口：

- 本地最新构建：`runs/final_results_dashboard/final_dashboard/index.html`
- 你当前浏览器里打开的旧入口：`server_artifacts/remote_result/report/index.html`

两者现在是同一份同步内容：每次重新运行最终看板生成后，都会自动同步到 `server_artifacts/remote_result/report`，保证你点开的都是最新版。

---

## 2. 页面功能总览

该页面用于展示**VT-Tiny-MOT**上的**个体异常检测（Individual）**与**群体异常检测（Group）**的最终对比结果，核心目标是把“论文里的模块结果”做成可视化证据面。

页面目前包含以下核心模块：

### 2.1 顶部控制

- 语言切换：中文 / English
- 任务切换：Individual / Group
- 方法选择：按任务筛选某个方法
- 序列选择：按检测序列筛选
- 视图模式切换：
  - 四画面对比（默认）：原图 / 热力图 / 轨迹 / 热力+轨迹
  - 单画面模式：仅保留一张画布，支持选择子层（原图/热力/轨迹/热力+轨迹）
- 自动播放：播放条与速度控制
- 热力参数：透明度、时间窗

### 2.2 统计卡片

实时显示当前任务/方法的：

- 方法总数（Method Count）
- 当前序列标注总数（Label Count）
- 当前序列异常标注数（Anomaly Count）
- 当前方法 AUROC

### 2.3 左侧侧边表格

- **Leaderboard**：按 AUROC 排序的指标排行，含 AUROC / AUPRC / F1 / P@K / R@K 等
- **Anomaly 类型**：不同异常类型（如速度异常、轨迹偏移、群体变化）在 Top-K 下的命中分布
- **Cases**：每个方法的 TP/FP/FN 展示
- **Method Status**：方法归属、任务角色、学习范式、是否主模型

### 2.4 主播放区（重点）

支持 **Individual / Group** 两个任务的轨迹可视化：

- 4 类画面同时展示（默认）：
  1. 原视频（轨迹叠加在原图）
  2. 热力图
  3. 轨迹图
  4. 热力+轨迹
- 点击轨迹可选中 track
- 显示当前分数、标签、轨迹评分排名、异常段落、群体关系（仅 Group）

### 2.5 右侧分析面板

支持三类内容切换：

- 方法解读：解释当前方法在该任务下的得分来源
- 轨迹洞察：当前被选中轨迹的分数/分解
- 指标明细：异常类型与 case 细节

---

## 3. 交互说明（你最常用的）

### 3.1 切任务（Individual / Group）

- Individual：显示单目标轨迹与单目标分数
- Group：显示群体轨迹以及群体关系（质心、邻域半径与连接线）

### 3.2 方法切换

- 每个方法保留自己的 score 行、分解信息与 case 统计
- 列表默认按 AUROC 降序

### 3.3 序列切换

- 每个序列会带有该任务下的异常标注统计
- 播放器会自动聚焦该序列的可视化帧区间

### 3.4 选中轨迹

- 在任一画布点击轨迹路径即可选中
- 选中后可查看：
  - 当前方法分数
  - 各任务分数
  - 标签与异常段
  - 分解分数（如 speed_profile / motion_profile / route_profile / group_profile 等）

---

## 4. 数据输入和字段映射

页面前端完全由两类 JSON 驱动：

1. `assets/final_dashboard_data.json`
   - 方法、任务、指标、cases、类型统计、方法标签
2. `assets/final_playback_data.json`
   - 每个序列对应的轨迹、背景帧、score/label payload、分解特征、可播放帧范围

### 4.1 任务级统计含义

- `stats.sequence_sample_count`：该任务该序列中的样本条数（按 labels）
- `stats.sequence_anomaly_count`：该任务该序列中的异常样本条数（label = 1）
- `stats.frame_start / frame_end`：播放帧范围
- `stats.visualized_tracks`：当前序列实际可展示轨迹数

### 4.2 轨迹条目（Track）常用字段

- `sample_id`, `track_id`, `sequence`
- `task_scores[task][method]`：各方法分数
- `task_labels[task]`：当前任务标签（正负样本、异常类型、窗口）
- `task_score_rows[task][method]`：原始 score 记录（用于追踪分解）
- `task_score_decomposition`：可解释字段（各子模型组件分数）
- `task_segments`：该任务下当前轨迹的异常段/标签段

---

## 5. 运行与重建命令

```powershell
python .\code\system\run_fusiontrack.py \
  --final-results-root server_artifacts/final_results_20260521 \
  --individual-label-file server_artifacts/fusiontrack_val_results_20260521/fusiontrack_val/protocol/individual_labels_val.jsonl \
  --group-label-file server_artifacts/fusiontrack_val_results_20260521/fusiontrack_val/protocol/group_labels_val.jsonl \
  --score-search-root server_artifacts/fusiontrack_val_results_20260521 \
  --score-search-root server_artifacts/fusiontrack_official_runs_tsad_20260521 \
  --score-search-root server_artifacts/final_results_20260522/fusiontrack_final_results_20260522_archive/official \
  --fused-jsonl server_artifacts/fusiontrack_val_results_20260521/fusiontrack_val/protocol/fused_trajectories_val.jsonl \
  --data-root data/VT-Tiny-MOT \
  --work-root runs/final_results_dashboard \
  --split val \
  --top-sequences 7 \
  --top-k 100 \
  --case-limit 12
```

说明：命令执行后，产物会出现在：

- `runs/final_results_dashboard/final_dashboard/`
- 同步镜像到：`server_artifacts/remote_result/report/`

---

## 6. 与论文模块映射（你做答辩时可以直接讲）

- 方法排行与指标对齐：`leaderboard`
- 任务分离：Individual / Group
- 异常类型统计：`anomaly type` 面板
- 轨迹级证据：播放区 + 点击轨迹 + 分解得分（支撑方法可解释性）
- 基线对照：不同方法间切换、TP/FP/FN case 对照

---

## 7. 当前版本后续优化建议（可选）

1. 增加导出 MP4 的批处理按钮（当前可通过外部脚本把 canvas 录制为视频）
2. Group 模式下增加“邻域半径阈值”的滑动条
3. 增加“方法对比固定帧”与“异常窗口快照”小图集
4. 增加一个“论文版导出摘要”按钮（自动导出 Top-1/Top-3 方法对比表）

---

## 8. 已知限制

- 部分旧文件中的中文注释/字符串是按 UTF-8 保存；在部分环境下显示可能出现乱码，建议统一在浏览器 UTF-8 下查看 HTML。
- `server_artifacts/` 与 `runs/` 在仓库中有 `.gitignore`，因此默认不纳入版本控制；若要把网页静态资源公开到仓库主页，需要额外把产物目录单独添加为 release/artifact 或解除忽略。
