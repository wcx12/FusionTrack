# FusionTrack 展示层模块覆盖清单（论文流程映射版）

> 版本：v3  
> 日期：2026-05-24  
> 目的：对照论文系统流程，说明当前展示层已经覆盖哪些模块，以及哪些模块仍需要后续算法产物继续增强。

## 一、论文模块到展示层映射

| 论文模块 | 当前展示位置 | 覆盖度 | 说明 |
|---|---|---|---|
| 数据与异常协议治理 | 协议概览、帮助弹窗、数据流审计 | 覆盖 | 页面明确说明 VT-Tiny-MOT 原始数据没有异常 label，Individual/Group 标签来自 synthetic anomaly injection。 |
| 数据抽取与预处理 | 数据流审计页、pipeline manifest、playback JSON | 主要覆盖 | 已展示可播放序列、轨迹、帧跨度、背景帧资源、任务 label/score 覆盖，并新增序列级 RGB/thermal/fused 覆盖率、缺失模态点数、背景状态和平均模态偏移。暂未展示每个底层抽取步骤耗时。 |
| 多模态融合轨迹 | 四画面播放、轨迹层、热力+轨迹层 | 覆盖 | 支持原视频、热力、轨迹、叠加四画面同步播放。 |
| Individual 分支 | 子模块证据区 route/speed/shape + 逐帧曲线 | 主要覆盖 | route 显示路径偏移曲线，speed 显示逐帧速度曲线，shape 显示局部转角曲线。后续可替换为算法原始 residual。 |
| Group 分支 | 群体关系面板、群体中心/半径、Group 事件聚合卡 | 主要覆盖 | 已能展示邻居数量、群体半径、关系连线，并把跨轨迹异常段聚合为事件卡。 |
| 个体-群体-事件融合 | 分数分量条形图 | 覆盖 | 已展示 `S_ind`、`S_grp`、`S_event`、`S_fused`。 |
| 事件级解释链 | 事件时间线、Group event cards | 主要覆盖 | 已并列展示真实异常段和预测段；Group 下增加涉及轨迹数、持续帧和事件中心。 |
| Registration 配准模块 | `Registration` 任务、配准流程、误差面板、3D 投影诊断 | 主要覆盖 | 已支持无 label registration task，展示旋转误差、平移误差、Chamfer、耗时、成功率、高误差案例和 source/reference/aligned 投影预览。 |
| 指标与排行榜 | 方法排名、类型分析、案例表、算法接入 | 覆盖 | 检测任务展示 AUROC/AUPRC/F1/P@K/R@K；配准任务展示 success rate 和误差指标。 |
| 可视化与交互 | 四画面播放、单画面切换、点击轨迹、语言切换 | 覆盖 | 默认四画面对比，可切换单画面，支持中文/英文。 |
| 部署/复现实验 | `run_fusiontrack.py`、report 同步、README、`.gitignore`、GitHub Pages | 覆盖 | 本地静态页可复现，`gh-pages` 已部署公开网页；`docs/` 已从 Git 跟踪中移除并由 `.gitignore` 忽略。 |

## 二、当前展示层已经包含的能力

- 顶部任务/方法/序列选择。
- 中英文切换。
- 统计卡片：方法数、标签/样本数、异常/失败数、当前指标。
- 协议概览：Individual、Group、Registration 三类任务解释。
- 四画面动态播放：原视频、热力图、轨迹、热力+轨迹。
- 单画面模式：轨迹、热力、热力+轨迹可切换。
- 播放控制：播放、暂停、帧滑条、热力透明度、时间窗口、播放速度。
- 点击轨迹选择目标。
- 高风险轨迹列表。
- 轨迹解释：分数、异常类型、帧段、速度、位移、运动长度。
- Individual 子模块曲线：route/speed/shape。
- Group 证据：邻居数量、群体半径、关系连线、跨轨迹事件聚合。
- Registration 证据：旋转误差、平移误差、Chamfer、耗时、成功/失败、3D 投影预览。
- 方法流程：检测任务和配准任务分别显示流程步骤。
- 分数分量：检测任务显示 `S_ind/S_grp/S_event/S_fused`，配准任务显示误差分量。
- 事件时间线：检测任务显示 GT segment 与 predicted segment。
- 数据流审计：展示 sequence、track、frame、background、label/score coverage，并包含序列级 RGB/thermal/fused 覆盖率、缺失模态点、背景状态和模态偏移。
- 实验分析：leaderboard、异常类型/配准指标、case、method status。
- GitHub Pages：公开静态页面已部署到 `https://wcx12.github.io/FusionTrack/`。

## 三、仍需后续实验或部署决策补强的部分

1. **底层步骤耗时与失败原因**
   - 当前页面已经能审计最终数据覆盖、模态覆盖和背景状态。
   - 后续如果要写成更完整的工程系统，还需要 pipeline 在 manifest 中输出 step-level runtime、失败原因、重试次数和各阶段输入输出文件校验。

2. **算法原始中间量**
   - 当前 Individual 曲线由轨迹点动态计算，可以用于展示与解释。
   - 如果后续算法输出逐帧 route deviation、speed residual、shape residual，应替换为算法原始中间量。

3. **真实学习式 Registration 3D**
   - 当前 3D 投影由 benchmark 误差字段生成，用于展示闭环和页面接口。
   - 后续 MPS-GAF 学习式模型输出真实 source/reference/aligned 点云后，应替换为真实点云渲染。

4. **自动化发布策略**
   - 公开页面已经通过 `gh-pages` 静态产物发布。
   - 后续可选择补 GitHub Actions，把报告生成和 Pages 发布从手动同步升级为 CI 自动发布。

## 四、结论

当前展示层已经覆盖论文系统的主要流程：数据协议、融合轨迹、Individual、Group、Fusion、Event、Registration、指标分析、交互可视化、序列级数据审计和公开静态部署。仍未完全闭环的内容主要依赖后续算法产物，尤其是算法原始 residual、学习式配准权重和真实点云中间结果。
