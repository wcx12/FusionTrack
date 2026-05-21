# FusionTrack Final Experiment Settings

This file is the human-readable version of `final_experiment_settings.json`.
It fixes the validation experiment protocol before the next server rerun.

## Common Protocol

| Item | Setting |
| --- | --- |
| Dataset | VT-Tiny-MOT |
| Source split | `train` |
| Sequence validation ratio | `0.2` |
| Main seed | `42` |
| Optional robustness seeds | `42, 43, 44` |
| Individual anomaly fraction | `0.1` |
| Group anomaly fraction | `0.1` |
| Group window size / stride | `16 / 8` |
| Individual evaluation key | `sample_id` |
| Individual key uniqueness | required in main protocol |
| Individual score coverage | exact label-score key match required |
| Group main evaluation key | `sample_id + window_id` |
| Group main key uniqueness | required in main protocol |
| Group main score coverage | exact label-score key match required |
| Group appendix diagnostic | `sample_id` any-window aggregation |
| Rank metric | `P@100 / R@100` |

## Fairness Rules

1. All methods in a task use the same generated protocol files.
2. Test split is not used for hyperparameter selection.
3. Validation scores cannot be used to choose per-method hyperparameters unless the run is labeled as tuning or ablation.
4. Paper baseline rows must come from official or paper-linked source code and record URL, commit, license, adapter, environment, and run manifest.
5. Local proxy methods are internal/proxy/ablation rows, not paper main-table baselines.
6. Epoch counts are not forced to be equal across unrelated algorithms. Instead, each method starts from a pre-declared method-appropriate budget, and every run records epochs, batch size, learning rate, window size, seed, adapter notes, and convergence diagnostics.
7. A deep-model result is final only after the convergence policy below is checked. A fixed epoch count by itself is not evidence of convergence.
8. Individual main-protocol evaluation must have one label row and one score row per `sample_id`; duplicate keys, missing scores, and extra scores fail the run.
9. Group main-protocol evaluation must have one label row and one score row per `sample_id + window_id`; duplicate keys, missing scores, and extra scores fail the run. The old `sample_id` any-window aggregation is appendix-only.

## Convergence Policy

The epoch numbers below are initial adapter budgets, not a guarantee that training has converged.

For final paper tables:

1. Use the method's official validation protocol when it exists. Otherwise split only the normal training data into train/validation. Do not use anomaly labels from the evaluation split to select epochs.
2. Save `loss_history.json`, `best_epoch`, `final_epoch`, `early_stop_reason`, GPU name, wall time, and the full `run_manifest.json`.
3. Use early stopping when the runner supports validation loss:
   - monitor: validation loss
   - patience: 5 epochs
   - min_delta: 0.001 relative improvement
   - restore best checkpoint
4. If no validation loss is available, use training loss plus score-stability diagnostics and report this limitation explicitly.
5. Start with a short pilot run, then allow up to 50 epochs for deep baselines. Stop early only when the monitored loss plateaus. If a model is still improving at 50 epochs, mark it `max-budget-not-converged` or extend the budget before putting it in the main table.

Every deep baseline in the main table must have one of these statuses:

| Status | Meaning |
| --- | --- |
| `converged` | Loss plateaued under the rule above |
| `early-stopped` | Validation loss selected a best checkpoint before the cap |
| `max-budget-not-converged` | The budget ended while the curve was still improving; this cannot be presented as a fully converged result |

## Local Benchmark Methods

| Method | Task | Training budget |
| --- | --- | --- |
| `fusiontrack_individual_nn` | individual | no epoch, `n_neighbors=1` |
| `fusiontrack_individual_context` | individual | no epoch, `n_neighbors=1` |
| `individual_iforest` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `individual_lof` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `individual_ocsvm` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `group_prediction_linear` | group | no epoch |
| `group_iforest` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_lof` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_ocsvm` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_temporal_graph_ae_proxy` | group | no deep epoch, `n_components=3`, `seed=42` |
| `fusiontrack_group_graph` | group | no epoch, `k_neighbors=3`, `rho_p=80`, `rho_v=20`, `eta=0.5` |

## Official Paper Baselines

| Method | Task | Epochs | Batch | LR | Window |
| --- | --- | ---: | ---: | ---: | --- |
| CETrajAD | individual | 10 | 64 | `1e-4` | variable trajectory |
| LM-TAD | individual | 5 | 64 | `5e-4` | token sequence |
| Pi-DPM | individual | 20 | 64 | `1e-3` | flattened trajectory |
| Anomaly Transformer | individual | 8 | 128 | `1e-4` | 64 |
| Anomaly Transformer | group | 8 | 256 | `1e-4` | 16 |
| DCdetector | individual | 8 | 128 | `1e-4` | 64 |
| DCdetector | group | 8 | 256 | `1e-4` | 16 |
| TranAD | individual/group | 5 | 128/256 | `1e-4` | 10 |

These are initial budgets from our current external runners and previous run manifests. They are not final convergence claims. Before any deep baseline enters the main table, rerun it with the convergence policy above and record the resulting status.

TranAD uses the official default-style `n_window=10`. If we need a fully standardized TSAD window comparison, run an appendix variant with individual window `64` and group window `16` before reporting that comparison.

## Server Execution

Full server runs should use tmux:

```bash
USE_TMUX=1 \
TMUX_SESSION=fusiontrack_val \
MODE=val \
GPU_ID=0 \
SEED=42 \
DATA_ROOT=/root/FusionTrack/data/VT-Tiny-MOT \
OUTPUT_ROOT=/root/autodl-tmp/fusiontrack_val_strict/protocol \
RESULT_ROOT=/root/autodl-tmp/fusiontrack_val_strict/results \
bash code/anomaly_detection/benchmark/runners/run_server_gpu_experiments.sh
```

Attach after reconnecting:

```bash
tmux attach -t fusiontrack_val
```

Logs are written under `logs/` by default, or under `LOG_ROOT` if that environment variable is set.

## Required Reruns

1. Regenerate group labels with the strict `sample_id + window_id` protocol.
2. Rerun all local group baselines under the strict group key.
3. Rerun or reconvert official group baselines so every score row carries top-level `window_id`.
4. Keep old `sample_id`-only group results only as appendix any-window diagnostics.
5. Regenerate individual and group matrix configs with `require_unique_keys=true` and `require_score_key_match=true`, then rerun any experiment whose metrics show duplicate keys, missing score keys, or extra score keys.
