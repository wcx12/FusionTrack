# Group Anomaly Detection

This directory is the stable entrypoint for FusionTrack group anomaly detection.
The implementation is shared with the benchmark package so that experiments,
tests, and the final dashboard all use the same code path.

## Entry Points

- `scoring.py`: object-level group anomaly scores for each group window.
- `graph.py`: group graph construction, relative motion features, and connected components.
- `tracking.py`: frame-level group discovery and temporal split/merge tracking.
- `temporal.py`: temporal KNN and hybrid FusionTrack group scoring.
- `run_group_method.py`: command-line wrapper for scoring JSONL group windows.

The underlying implementation lives in:

```text
code/anomaly_detection/benchmark/fusiontrack/
```

The baseline group methods live in:

```text
code/anomaly_detection/benchmark/baselines/
```

## CLI Usage

Run from the repository root:

```bash
python code/anomaly_detection/group/run_group_method.py input_windows.jsonl output_scores.jsonl
```

Optional parameters:

```bash
python code/anomaly_detection/group/run_group_method.py input_windows.jsonl output_scores.jsonl \
  --k-neighbors 3 \
  --rho-p 80 \
  --rho-v 20 \
  --eta 0.5
```

## Python Usage

```python
from group import score_group_windows

rows = score_group_windows(group_windows, k_neighbors=3, rho_p=80.0, rho_v=20.0)
```

If importing from outside this repository, add `code/anomaly_detection` to
`PYTHONPATH` first.

## Why This Directory Exists

Earlier development placed the research implementation under the benchmark
tree because the same code had to support protocol generation, ablation
experiments, and metric evaluation. This directory keeps a clean public-facing
group module while preserving that tested benchmark implementation.
