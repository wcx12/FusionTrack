# FusionTrack 系统完成度清单

本文档用于说明当前系统已经完成什么、还缺什么，以及后续继续实现时应优先处理哪些模块。状态来源见 `system_completion_status.json`。

## 状态定义

- `done`：当前代码、文档和可验证产物已经形成闭环。
- `partial`：已有可展示或可运行版本，但距离论文理想系统仍有明确缺口。
- `pending`：尚未形成可用实现或尚未接入最终展示链路。

## 当前结论

FusionTrack 当前已经形成一版可展示、可部署、可解释、可导出的论文系统闭环：可以展示 Individual、Group、Registration 三类任务，支持动态可视化、中英文切换、数据流审计、解释证据、导出和 GitHub Pages 公开访问。

但它还不是最终论文理想版本。主要缺口集中在算法层：真实学习式个体/群体异常检测结果、官方深度 baseline 多种子收敛结果、Registration 真实 source/reference/aligned 点云输出，以及最终验收材料中的运行状况和验收结论。

## 模块清单

| 模块 | 状态 | 当前情况 | 后续要做 |
| --- | --- | --- | --- |
| 数据接入与数据流审计 | done | 已生成 dataset manifest、媒体类型、背景帧和任务/序列审计。 | 真实部署时确认 `data_root` 指向完整数据集。 |
| 异常标签协议 | partial | Individual/Group 使用 synthetic anomaly injection，页面已展示协议来源。 | 有真实异常标注后按 key policy 接入并重新评估。 |
| 个体异常检测 | partial | 已接入分数、事件证据、解释 schema 和四画面对比。 | 用最终学习式模型和官方 baseline 多种子结果替换/增强当前结果。 |
| 群体异常检测 | partial | 已有群体窗口、群体关系、事件聚合和可视化展示。 | 补强学习式群体异常模型、消融实验和主表结果。 |
| 点云配准诊断 | partial | 已支持 Registration 动态点云视图、误差面板和点云来源标注。 | 当前最终数据仍是 `synthetic_preview`，需要接入真实点云输出。 |
| 最终可视化网页 | done | 已支持中英文、展示/审计模式、四画面对比、Registration 视图和丰富交互。 | 后续每次算法结果更新后重新生成并部署页面。 |
| 解释证据与导出交付 | done | 已支持解释 schema、事件时间线、PNG/JSON/CSV/ZIP 导出。 | 报告 ZIP 较大，后续可改成 GitHub Release 附件。 |
| GitHub Pages 公开部署 | done | `gh-pages` 已推送新版页面和数据资源。 | 控制 history 归档体积，避免重复提交整套背景帧。 |
| 验收与答辩材料 | partial | 已有功能说明和答辩常见问题。 | 软件运行状况和验收结论按用户要求仍留空，最终答辩前补齐。 |

## 下一步优先级

1. **补算法证据**：优先把真实学习式个体/群体异常模型、官方深度 baseline 多种子结果接入最终结果表。
2. **补真实配准点云**：让 MPS-GAF 或外部学习式配准 baseline 输出 `registration_points.source/reference/aligned`，替换当前 `synthetic_preview`。
3. **重新生成公开页面**：每次实验结果变化后，用 `code/system/configs/final_dashboard.local.example.json` 重新构建 dashboard 并发布 Pages。
4. **补验收材料**：最终确认运行截图、公开网页地址、测试结果和百分制验收结论后，再填写软件运行状况和验收结论。

## 答辩表述建议

当前系统可以表述为：“论文系统的一版完整展示闭环实现，已经覆盖数据接入、异常协议、个体/群体异常展示、配准诊断、结果汇总、动态可视化、解释证据、导出和公开部署。算法层仍在持续替换为更强的学习式模型，当前页面可以无缝接入后续模型输出。”
