# FusionTrack 当前版本快照（用于回滚与恢复）

## 1. 基础回滚信息

| 项目 | 值 |
|---|---|
| 分支 | `main` |
| 当前提交 | `068ce6781679831eeee06472eab3d974affd34ae` |
| 最后提交标题 | `docs: add system completion checklist for full system planning` |
| 远端仓库 | `https://github.com/wcx12/FusionTrack.git` |
| 默认远端分支 | `main` |
| 记录时间 | `2026-05-23` |

## 2. 当前工作区状态（未提交变更）

已修改：
- `code/registration/README.md`

未跟踪：
- `code/registration/non_learning_baselines.py`
- `code/registration/non_learning_benchmark_plan.md`
- `code/registration/run_registration_benchmark.py`
- `code/registration/run_registration_benchmark_suite.py`

说明：这部分为当前实验/注册模块扩展文件，未纳入本次提交，如果需要保留请单独提交；需要回滚到本次快照时可忽略它们以恢复到主干清洁状态。

## 3. 关键里程碑提交

- `370d554`：完善 dashboard 说明与分组洞察
- `04fad80`：修复组播数据加载和播放链路
- `ff63d0a`：中文 README 补齐
- `b5dea62`：合并 anomaly benchmark 与 dashboard（完整实验结果展示阶段）
- `068ce67`：系统清单与能力差距整理（本快照）

## 4. 推荐回滚命令

恢复到本快照提交（保留未提交文件不变）：
```bash
git switch main
git pull origin main
git reset --hard 068ce6781679831eeee06472eab3d974affd34ae
```

若需要连未提交文件一起恢复（谨慎）：
```bash
git clean -fd code/registration
```

## 5. 与全链路打通相关的当前起点

已具备的入口文件：
- `code/system/fusiontrack/final_results.py`
- `code/system/fusiontrack/final_dashboard.py`
- `system_completion_checklist.md`
- `docs/superpowers/plans/2026-05-23-fusiontrack-full-system.md`
- `README.md`

建议按下步任务开启：
1. 构建 run manifest（记录每次跑分的 seed/epoch/参数/代码 commit）
2. 统一方法注册（`owner/role/family/learning_type`）
3. 将 individual/group 事件分量加入可视化 payload
4. 连接事件分量与播放面板解释
5. 加入可复现输出与导出
