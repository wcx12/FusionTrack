# VT-Tiny-MOT 异常检测 Benchmark

本目录用于建设 FusionTrack 在 VT-Tiny-MOT 上的可复现异常检测 benchmark。VT-Tiny-MOT 是 RGB/Thermal 多目标 tracking 数据集，不自带异常检测标签；本项目会固定异常注入和评价协议，使 baseline、FusionTrack 变体和最终融合方法在同一数据划分、同一标签文件和同一指标下比较。

## 目标

- 固定 train/val/test 序列划分，避免同一场景泄漏到不同 split。
- 在验证集和测试集上生成可复现的个体级与群体级异常注入标签。
- 统一所有方法的分数文件 schema、评价指标和报告格式。
- 为论文结果保留可追溯的 metrics、manifest、运行日志和可视化输出。

## 阶段

1. Phase 0：建立远程执行安全边界和本目录局部忽略规则。
2. Phase 1：生成数据 split、异常注入清单和标签 schema。
3. Phase 2：实现统一评价模块和报告导出。
4. Phase 3-7：实现个体级 baseline、论文对齐的 FusionTrack 个体方法、群体级方法、群体 baseline 和最终融合。
5. Phase 8-9：整理远程运行脚本、实验输出、结果表、图和论文可用说明。

## 目录约定

- `protocol/`：后续存放 split、异常注入和标签 schema 相关代码。
- `evaluation/`：后续存放指标计算、分数读取和报告导出代码。
- `baselines/`：后续存放个体级与群体级 baseline。
- `fusiontrack/`：后续存放论文对齐的 FusionTrack 方法和融合逻辑。
- `runners/`：后续存放统一运行入口。
- `reports/`：后续存放结果表和可视化构建代码。
- `outputs/`、`runs/`、`checkpoints/`、`logs/`：仅用于本地或远程实验输出，已由本目录 `.gitignore` 忽略。

## 后续运行原则

- 远程服务器凭据只能由用户交互提供，不能写入仓库、脚本、日志、manifest 或 commit。
- 每次实验都必须记录 git commit、实际命令、seed、数据集路径、split 文件和输出路径。
- 测试集只用于配置冻结后的最终报告；阈值和超参数只能在验证集上选择。
- 所有输出文件应写入被忽略的实验目录，避免将模型权重、缓存或大体积结果提交到仓库。
