# 群体异常检测

这个目录是 FusionTrack 群体异常检测的稳定对外入口。实际算法实现与 benchmark 包共享，保证实验脚本、测试代码和最终可视化系统使用同一套代码路径。

## 入口文件

- `scoring.py`：对每个群体窗口输出 object-level 群体异常分数。
- `graph.py`：群体图构建、相对运动特征和连通分量计算。
- `tracking.py`：逐帧群体发现，以及跨帧 split/merge 事件追踪。
- `temporal.py`：群体 temporal KNN 与 hybrid FusionTrack 群体打分。
- `run_group_method.py`：面向 JSONL 群体窗口文件的命令行打分入口。

底层实现位于：

```text
code/anomaly_detection/benchmark/fusiontrack/
```

群体 baseline 位于：

```text
code/anomaly_detection/benchmark/baselines/
```

## 命令行用法

从仓库根目录运行：

```bash
python code/anomaly_detection/group/run_group_method.py input_windows.jsonl output_scores.jsonl
```

可选参数示例：

```bash
python code/anomaly_detection/group/run_group_method.py input_windows.jsonl output_scores.jsonl \
  --k-neighbors 3 \
  --rho-p 80 \
  --rho-v 20 \
  --eta 0.5
```

## Python 用法

```python
from group import score_group_windows

rows = score_group_windows(group_windows, k_neighbors=3, rho_p=80.0, rho_v=20.0)
```

如果从仓库外部导入，请先把 `code/anomaly_detection` 加入 `PYTHONPATH`。

## 为什么保留这个目录

早期开发时，研究实现主要放在 benchmark 目录下，因为同一套代码需要同时服务协议生成、消融实验、指标评估和最终网页展示。这个目录提供更清晰的群体异常检测入口，同时复用已经经过测试的 benchmark 实现，避免出现两套逻辑不一致的问题。
