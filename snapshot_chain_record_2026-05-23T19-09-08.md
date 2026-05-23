# FusionTrack 全链路打通快照（记录版）

- 生成时间：2026-05-23T19:09:08
- 分支：main
- HEAD：d4d556baff2f939286ecf32a31854e273550cb63
- 最近提交：d4d556b docs: add rollback snapshot and recovery notes

## 一、当前状态摘要

- 该快照用于 main 分支增量阶段；工作区尚有未提交改动。
- 目标是把“注册基线结果 -> 实验 manifest -> 可视化报告”打通为一条最短全链路。

### 未提交改动（git status --short）

M code/registration/README.md
M code/registration/requirements.txt
M code/system/fusiontrack/experiment_adapter.py
M code/system/fusiontrack/pipeline.py
M code/system/run_fusiontrack.py
?? code/registration/non_learning_baselines.py
?? code/registration/non_learning_benchmark_plan.md
?? code/registration/run_registration_benchmark.py
?? code/registration/run_registration_benchmark_suite.py
?? code/system/fusiontrack/registration_adapter.py
?? code/system/tests/test_registration_adapter.py
?? snapshot_before_full_pipeline_linking_2026-05-23T18-56-20.md
?? snapshot_full_chain_bootstrap_2026-05-23T19-30-00.md

### 关键改动清单

1. `code/registration/non_learning_baselines.py`
   - 非学习注册基线算法集合：identity / ICP / point-to-plane / trimmed ICP / RANSAC / FPFH+RANSAC。
2. `code/registration/run_registration_benchmark.py`
   - 基准脚本，输出 `baseline_summary.json`。
3. `code/registration/run_registration_benchmark_suite.py`
   - 批量 case 脚本，输出 `suite_summary.json`。
4. `code/registration/non_learning_benchmark_plan.md`
   - 方案说明与示例命令。
5. `code/registration/README.md` 与 `code/registration/requirements.txt`
   - 文档和依赖更新。
6. `code/system/fusiontrack/registration_adapter.py`
   - 注册结果映射适配器：baseline summary -> score jsonl + metrics + fused trajectories + experiment manifest。
7. `code/system/fusiontrack/pipeline.py`
   - 新增注册链路入口：`run_registration_experiment(...)`。
8. `code/system/fusiontrack/experiment_adapter.py`
   - manifest 文件路径解析增强，兼容 manifest 在 `registration_artifacts` 下引用 `registration_scores/*`。
9. `code/system/run_fusiontrack.py`
   - CLI 增加 `--registration-benchmark-summary` 与 `--registration-result-method`。
10. `code/system/tests/test_registration_adapter.py`
   - 新增链路级最小测试。

## 二、回滚命令

- 全量撤销未提交改动：
  - `git restore --worktree --staged .`
- 回到上次已知提交：
  - `git reset --hard d4d556b`
  - `git clean -fd code/registration code/system/fusiontrack snapshot_*.md`
- 仅回退新增文件：
  - `git clean -fd -- code/registration/non_learning_baselines.py code/registration/non_learning_benchmark_plan.md code/registration/run_registration_benchmark.py code/registration/run_registration_benchmark_suite.py code/system/fusiontrack/registration_adapter.py code/system/tests/test_registration_adapter.py`

## 三、链路复现验证（相对路径）

1. 先准备一份注册基准输出 `baseline_summary.json`。
2. 执行命令：
   - `python code/system/run_fusiontrack.py --data-root data/VT-Tiny-MOT --work-root runs/fusiontrack_v1 --split test --registration-benchmark-summary <baseline_summary> --registration-result-method icp_point_to_point`
3. 验证以下输出是否存在：
   - `runs/fusiontrack_v1/registration_artifacts/registration_experiment_manifest.json`
   - `runs/fusiontrack_v1/pipeline_summary_test_experiment.json`
   - `runs/fusiontrack_v1/pipeline_manifest_experiment_report_test.json`
   - `runs/fusiontrack_v1/report/index.html`

### 本轮新增链路自检（已执行）

- 使用命令（示例）：
  - `python code/system/run_fusiontrack.py --data-root data/VT-Tiny-MOT --work-root tmp_reg_chain_test2 --split test --registration-benchmark-summary tmp_reg_chain_test2/base_summary.json --registration-result-method icp_point_to_point --top-sequences 1`
- 成功输出：
  - 命令返回 JSON summary（`mode=experiment_report`）
  - `tmp_reg_chain_test2/registration_artifacts/registration_experiment_manifest.json`
  - `tmp_reg_chain_test2/pipeline_summary_test_experiment.json`
  - `tmp_reg_chain_test2/pipeline_manifest_experiment_report_test.json`
  - `tmp_reg_chain_test2/report/index.html`
- 注意：上面路径为演示用途；正式使用请改为你自己的 `--work-root` 与 baseline summary 文件。

## 四、注意事项

- 环境中的 pytest 在当前 Python 配置下可能会遇到 pyreadline-collections 兼容问题，可用 `python -m py_compile` 做语法验证。
- 本快照版本建议提交前再做一次清理（保留你认为需要的历史快照文件）。
