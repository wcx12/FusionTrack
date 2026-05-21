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
| `fusiontrack_individual_ensemble` | individual | no epoch, rank ensemble of nearest-feature, LOF novelty, and Isolation Forest components |
| `fusiontrack_individual_context` | individual | no epoch, `n_neighbors=1` |
| `individual_iforest` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `individual_lof` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `individual_ocsvm` | individual | no epoch, `contamination=0.05`, `seed=42` |
| `group_prediction_linear` | group | no epoch |
| `group_iforest` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_lof` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_ocsvm` | group | no epoch, `contamination=0.05`, `seed=42` |
| `group_temporal_graph_ae_proxy` | group | no deep epoch, `n_components=3`, `seed=42` |
| `fusiontrack_group_temporal_knn` | group | no epoch, `n_neighbors=3`, standardized group-feature KNN |
| `fusiontrack_group_hybrid` | group | no epoch, rank fusion of prediction residual, graph cohesion, and temporal-profile components |
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
| CATCH | individual/group | 20 | 256 | `1e-4` | 64 / 16 |
| CutAddPaste | individual/group | 20 | 256 | `1e-4` | 64 / 16 |
| TimeMixer | individual/group | 20 | 256 | `1e-4` | 64 / 16 |
| SensitiveHUE | individual/group | 20 | 256 | `1e-4` | 64 / 16 |

These are initial budgets from our current external runners and previous run manifests. They are not final convergence claims. Before any deep baseline enters the main table, rerun it with the convergence policy above and record the resulting status.

TranAD uses the official default-style `n_window=10`. If we need a fully standardized TSAD window comparison, run an appendix variant with individual window `64` and group window `16` before reporting that comparison.

The recent-baseline runner uses official external checkouts and imports only the official model/loss/augmentation modules. CATCH, CutAddPaste, and TimeMixer are top-venue rows. SensitiveHUE is tracked as a supplementary official-source candidate until a peer-reviewed venue/source record is verified.

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

## Completed Reruns

1. Regenerated strict validation protocol on the remote server for revision `b3b8599`.
2. Reran local individual and group benchmark matrices under strict key matching in `tmux` session `fusiontrack_group_knn_b3b8599`.
3. Added `fusiontrack_group_temporal_knn` to the generated group matrix and obtained full-coverage group scores.
4. Verified the generated individual and group summaries have zero duplicate keys, zero missing score keys, and zero extra score keys.
5. Reran official-source LM-TAD, Pi-DPM, TranAD, Anomaly Transformer, and DCdetector on the remote GPU under revision `b3b8599`.
6. Regenerated strict official individual and group summaries in `/root/autodl-tmp/fusiontrack_b3b8599_official_20260522`, with zero duplicate, missing, or extra score keys for all main-table official rows.
7. Reran CETrajAD and confirmed it remains coverage-failed: `770/829` score rows, `59` missing score keys.
8. Reran the max-budget official-source deep baselines with a 50-epoch cap in `/root/autodl-tmp/fusiontrack_b3b8599_convergence_20260522`. LM-TAD converged by validation loss; TranAD and Anomaly Transformer individual/group remained `max-budget-not-converged`.
9. Added and reran two enhanced FusionTrack rows in `/root/autodl-tmp/fusiontrack_b3b8599_methods_20260522`: `fusiontrack_individual_ensemble` and `fusiontrack_group_hybrid`, both under strict key matching.
10. Added recent official-source runner `run_recent_official_fusiontrack.py` and started strict validation runs for CATCH, CutAddPaste, TimeMixer, and SensitiveHUE in `/root/autodl-tmp/fusiontrack_recent_official_20260522`.

## Remaining Reruns

1. Keep CETrajAD out of the strict main table unless a full-coverage official-source scorer is implemented.
2. Extend remaining max-budget-not-converged deep runs beyond 50 epochs if the paper needs final convergence claims instead of reporting the current budget status.
3. Keep old `sample_id`-only group results only as appendix any-window diagnostics.
4. Rerun any experiment whose metrics show duplicate keys, missing score keys, or extra score keys.
5. Treat `fusiontrack_group_hybrid` as the current best validation candidate, but document that it uses rank-direction choices fixed in the method config and should be confirmed on an untouched test split or additional seeds before a final paper claim.
