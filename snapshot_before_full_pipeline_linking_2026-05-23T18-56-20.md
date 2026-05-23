# Full-workchain rollback snapshot

生成时间: ' + (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') + '

## 1) 当前分支与提交基线
' + (git rev-parse --abbrev-ref HEAD) + '

```bash
' + (git rev-parse HEAD) + '
' + (git log --oneline -n 1) + '
```

当前工作区相对 `origin/main` 前进：
```bash
' + (git status --short --branch) + '
```

## 2) 待提交改动清单

```text
' + (git status --short) + '
```

### pipeline.py 增量
- 已在 `code/system/fusiontrack/pipeline.py` 增加 run manifest 链路（manifest_path），用于实验可追溯。
  - run_smoke_pipeline
  - build_experiment_report
  - build_final_results_report
- manifest 包含 run_id / mode / timestamp / split / 路径上下文 / 执行输入参数摘要。

### registration 模块新增
- `code/registration/non_learning_baselines.py`：基于 NumPy/SciPy 的非学习基线集合。
- `code/registration/run_registration_benchmark.py`：基线批量评测脚本。
- `code/registration/run_registration_benchmark_suite.py`：多 case 套件脚本。
- `code/registration/non_learning_benchmark_plan.md`：基线实验计划与示例命令。
- `code/registration/README.md`：将绝对路径示例改为仓库相对路径，强化可复现性。

## 3) 备份与回滚命令（按需）

### 快速回滚当前改动（未提交工作区）
```bash
git restore --source=HEAD --worktree --staged --quiet .
```

### 回到快照前的提交节点
```bash
 git reset --hard d4d556b
```

### 若仅回到某一文件版本
```bash
git checkout -- code/system/fusiontrack/pipeline.py
git checkout -- code/registration/README.md
```

### 暂存并提交（建议先检查）
```bash
git add code/system/fusiontrack/pipeline.py code/registration/README.md code/registration/non_learning_baselines.py code/registration/non_learning_benchmark_plan.md code/registration/run_registration_benchmark.py code/registration/run_registration_benchmark_suite.py
git commit -m "feat: add run manifest and non-learning registration baselines"
```

### 若需重建快照标签
```bash
git tag -f snapshot/current_full_workflow_linking-$(Get-Date -Format yyyyMMdd_HHmmss)
```

## 4) 全链路打通建议（当前优先级）

1. 先把 `run_registration_*` 输出统一成 `FinalResults` 能读的 `score JSONL`（sample_id/score/sequence）
2. 在 `run_fusiontrack.py` 增加注册模块入口，和现有 `FinalResultsDashboard` 统一目录输出。
3. 产出一套最小可视化验证命令（本地 + 服务器）：
   - 生成 final_summary
   - 生成 final_dashboard
   - 同时输出 manifest + result 说明文件
4. 将该快照文件加入仓库或保留本地外部清单用于回滚。
