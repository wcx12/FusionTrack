# 群体异常检测

这个目录是 FusionTrack 群体异常检测的稳定对外入口。实际算法实现与 benchmark 包共享，保证实验脚本、测试代码和最终可视化系统使用同一套逻辑。

## 入口文件

- `scoring.py`：对每个群体窗口输出 object-level 群体异常分数。
- `graph.py`：群体图构建、相对运动特征和连通分量计算。
- `tracking.py`：逐帧群体发现，以及跨帧 split/merge 事件跟踪。
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

## 输出字段

`score_group_windows()` 每个 object-level score row 现在包含：

- `score`：当前对象在窗口内的最终群体异常分数。
- `component_scores`：`leave`、`motion`、`neighbor`、`count`、`dispersion`、`split_merge`、`object_group`、`group_event` 等分量。
- `event_score`：逐帧事件证据中的最大分数。
- `frame_event_scores`：逐帧事件证据序列，每个元素包含 `frame`、`score`、`dominant_reason` 和逐帧分量。
- `event_segments`：由逐帧分数合并得到的事件段，用于最终 dashboard 的事件时间线。

这些字段会继续进入系统展示层，支撑“群体结构变化在哪些帧发生、由哪个分量主导”的解释链路。

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
