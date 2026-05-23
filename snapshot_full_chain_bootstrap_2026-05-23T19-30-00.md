# 全链路改造快照（优先回滚记录）

- 生成时间：2026-05-23 19:01:40
- 仓库：$root
- 当前分支：$branch
- 当前提交：$head

## 最近提交（用于对照）

`	ext
d4d556b docs: add rollback snapshot and recovery notes 068ce67 docs: add system completion checklist for full system planning 370d554 Enhance dashboard explanations and group insights
`

## 与远端关系 / 工作区状态

`	ext
## main...origin/main [ahead 1]  M code/registration/README.md  M code/registration/requirements.txt  M code/system/fusiontrack/pipeline.py ?? code/registration/non_learning_baselines.py ?? code/registration/non_learning_benchmark_plan.md ?? code/registration/run_registration_benchmark.py ?? code/registration/run_registration_benchmark_suite.py ?? snapshot_before_full_pipeline_linking_2026-05-23T18-56-20.md
 M code/registration/README.md  M code/registration/requirements.txt  M code/system/fusiontrack/pipeline.py ?? code/registration/non_learning_baselines.py ?? code/registration/non_learning_benchmark_plan.md ?? code/registration/run_registration_benchmark.py ?? code/registration/run_registration_benchmark_suite.py ?? snapshot_before_full_pipeline_linking_2026-05-23T18-56-20.md
`

## 当前未提交变更

`	ext
 M code/registration/README.md  M code/registration/requirements.txt  M code/system/fusiontrack/pipeline.py ?? code/registration/non_learning_baselines.py ?? code/registration/non_learning_benchmark_plan.md ?? code/registration/run_registration_benchmark.py ?? code/registration/run_registration_benchmark_suite.py ?? snapshot_before_full_pipeline_linking_2026-05-23T18-56-20.md
`

## 变更清单

### 已修改文件

- code/system/fusiontrack/pipeline.py
  - 已有 run manifest（pipeline_manifest_*.json）的基础能力。
  - 已在 un_smoke_pipeline、uild_experiment_report、uild_final_results_report 写入摘要与 manifest。
  - 注意：uild_experiment_report 已可直接接收实验 manifest 并输出报告。

- code/registration/README.md
  - 已将示例路径改为仓库相对路径，减少绝对路径依赖。

- code/registration/requirements.txt
  - 已补齐非学习基线脚本的依赖项（
umpy, scipy, h5py, 	orch）。

### 新增文件

- code/registration/non_learning_baselines.py
  - 非学习注册基线实现（identity、ICP、point-to-plane、trimmed ICP、RANSAC、FPFH+RANSAC）。
- code/registration/run_registration_benchmark.py
  - 非学习基线批量评测入口，输出 aseline_summary.json。
- code/registration/run_registration_benchmark_suite.py
  - 多配置批量脚本，输出 suite_summary.json。
- code/registration/non_learning_benchmark_plan.md
  - 试验说明与建议命令。
- snapshot_before_full_pipeline_linking_2026-05-23T18-56-20.md
  - 上一版回滚记录。

## 回滚建议（优先）

### 1. 快速回退到当前工作区未改动状态（保留提交历史）

`ash
git restore --worktree --staged .
`

### 2. 回退到上一个已知提交（仅在确认本地改动全部废弃时）

`ash
git reset --hard d4d556b
`

### 3. 只回退某个文件

`ash
git restore -- code/system/fusiontrack/pipeline.py

git restore -- code/registration/README.md
`

### 4. 将当前快照版本打标签（便于后续跳回）

`ash
git tag -f snapshot/fullchain-pre-link-2026-05-23-$((Get-Date).ToString('yyyyMMdd_HHmmss'))
`

## 全链路打通建议（下一步执行顺序）

1. 建议先新增 pipeline 侧的注册基线转换入口（将 un_registration_*.json 映射到统一 experiment manifest）。
2. 将该入口挂到 un_fusiontrack.py，形成 --registration-benchmark-summary 的一键路径。
3. 增加最小自检脚本，输出：manifest、pipeline_summary、manifest 文件是否存在。
4. 如需继续可视化，补充将 sample_id 映射为伪 sequence/track，输出可回放页。

