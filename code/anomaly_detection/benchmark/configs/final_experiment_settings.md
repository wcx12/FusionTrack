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
| `fusiontrack_individual_ensemble_calibrated` | individual | no epoch, same rank ensemble plus feature-stratified rank calibration over `mean_speed`, `duration_frames`, and `num_points` |
| `fusiontrack_individual_ensemble_tuned_auprc` | individual | validation-tuned score-grid candidate, weights `0.45/0.45/0.10`, motion calibration global weight `0.3` |
| `fusiontrack_individual_ensemble_tuned_topk` | individual | validation-tuned score-grid candidate, weights `0.60/0.30/0.10`, motion calibration global weight `0.3` |
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
| `fusiontrack_group_hybrid_gated` | group | no epoch, same hybrid components plus residual gate that suppresses graph/temporal side evidence when prediction residual rank is low |
| `fusiontrack_group_hybrid_tuned_auroc_topk` | group | validation-tuned score-grid candidate, ungated weights `0.50/0.25/0.25` |
| `fusiontrack_group_hybrid_tuned_auprc_f1` | group | validation-tuned score-grid candidate, ungated weights `0.60/0.30/0.10` |
| `fusiontrack_group_hybrid_tuned_fine_auprc` | group | fine validation-tuned candidate, ungated weights `0.47/0.41/0.12` |
| `fusiontrack_group_hybrid_tuned_fine_topk` | group | fine validation-tuned candidate, ungated weights `0.45/0.43/0.12` |
| `fusiontrack_group_hybrid_tuned_fine_f1` | group | fine validation-tuned candidate, ungated weights `0.46/0.42/0.12` |
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
7. Reran CETrajAD original adapter and confirmed the initial failure mode: `770/829` score rows, `59` missing score keys.
8. Reran the max-budget official-source deep baselines with a 50-epoch cap in `/root/autodl-tmp/fusiontrack_b3b8599_convergence_20260522`. LM-TAD converged by validation loss; TranAD and Anomaly Transformer individual/group remained `max-budget-not-converged`.
9. Added and reran two enhanced FusionTrack rows in `/root/autodl-tmp/fusiontrack_b3b8599_methods_20260522`: `fusiontrack_individual_ensemble` and `fusiontrack_group_hybrid`, both under strict key matching.
10. Added recent official-source runner `run_recent_official_fusiontrack.py` and started strict validation runs for CATCH, CutAddPaste, TimeMixer, and SensitiveHUE in `/root/autodl-tmp/fusiontrack_recent_official_20260522`.
11. Reran CETrajAD with a full-coverage FusionTrack adapter in `/root/autodl-tmp/fusiontrack_cetrajad_fullcoverage_20260522`; strict individual evaluation now has `829/829` scores, zero duplicate keys, zero missing score keys, and zero extra score keys. The result is `official_cetrajad_fullcoverage` with AUROC `0.521092`, AUPRC `0.106465`, F1 `0.193437`, P@100 `0.080000`, R@100 `0.096386`, and convergence status `no-loss-history`.
12. Added `fusiontrack_individual_ensemble_calibrated` and `fusiontrack_group_hybrid_gated`, then reran strict validation in `/root/autodl-tmp/fusiontrack_improved_methods_v2_20260522`. The predeclared individual calibrated row improved to AUROC `0.625052`, AUPRC `0.160261`, F1 `0.280899`, P@100 `0.160000`, R@100 `0.192771`. The predeclared group gated row did not improve over the old hybrid.
13. Ran a validation score-grid over cached FusionTrack components in `/root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/score_grid_fast`. Best individual AUPRC is `0.166826` with weights `0.45/0.45/0.10` and motion calibration global weight `0.3`; best group AUROC/P@100 is `0.708855`/`0.160000` with ungated weights `0.50/0.25/0.25`; best group AUPRC/F1 is `0.092200`/`0.215827` with ungated weights `0.60/0.30/0.10`.
14. Reproduced the four validation-tuned candidates with the standard strict matrix runner in `/root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/tuned_subset`. Results: `fusiontrack_individual_ensemble_tuned_auprc` AUROC `0.624326`, AUPRC `0.166826`, F1 `0.272727`, P@100 `0.190000`, R@100 `0.228916`; `fusiontrack_individual_ensemble_tuned_topk` AUROC `0.618915`, AUPRC `0.162399`, F1 `0.255639`, P@100 `0.200000`, R@100 `0.240964`; `fusiontrack_group_hybrid_tuned_auroc_topk` AUROC `0.708855`, AUPRC `0.082023`, F1 `0.198473`, P@100 `0.160000`, R@100 `0.181818`; `fusiontrack_group_hybrid_tuned_auprc_f1` AUROC `0.672912`, AUPRC `0.092200`, F1 `0.215827`, P@100 `0.150000`, R@100 `0.170455`.
15. Ran a finer group weight search and reproduced three fine validation-tuned candidates in `/root/autodl-tmp/fusiontrack_improved_methods_v2_20260522/tuned_subset/results/group_fine`. Results: `fusiontrack_group_hybrid_tuned_fine_auprc` AUROC `0.680515`, AUPRC `0.098513`, F1 `0.215569`, P@100 `0.180000`, R@100 `0.204545`; `fusiontrack_group_hybrid_tuned_fine_topk` AUROC `0.679195`, AUPRC `0.098164`, F1 `0.216867`, P@100 `0.190000`, R@100 `0.215909`; `fusiontrack_group_hybrid_tuned_fine_f1` AUROC `0.679892`, AUPRC `0.097720`, F1 `0.218182`, P@100 `0.180000`, R@100 `0.204545`.
16. Added train-to-test holdout protocol and multi-seed aggregation runners, then confirmed local/proxy methods on untouched VT-Tiny-MOT `test` over seeds `42,43,44`. Combined output is `/root/autodl-tmp/fusiontrack_holdout_multiseed_combined_20260522`, archived locally at `server_artifacts/final_results_20260522/fusiontrack_holdout_multiseed_combined_20260522_summaries.tar.gz`. Strict alignment has `78/78` metric rows with zero duplicate, missing, or extra keys. Key mean/std results: individual `fusiontrack_individual_ensemble_tuned_auprc` AUROC `0.645935+/-0.031040`, AUPRC `0.190092+/-0.021638`, F1 `0.274699+/-0.031964`, P@100 `0.246667+/-0.020817`, R@100 `0.154167+/-0.013010`; individual `individual_lof` AUPRC is marginally higher at `0.191153+/-0.042912`; group `fusiontrack_group_hybrid_tuned_fine_topk` AUPRC `0.091499+/-0.021386`, and group `fusiontrack_group_hybrid_tuned_auroc_topk` AUROC `0.794720+/-0.018227`, both well above `group_prediction_linear` AUPRC `0.017417+/-0.003255` and AUROC `0.637508+/-0.008517` under this holdout protocol.

## Remaining Reruns

1. CETrajAD can be reported as `official_cetrajad_fullcoverage` in the strict individual table, but the paper must note the full-coverage adapter, `coordinate_scale=1.0`, and `no-loss-history` status; keep the original `770/829` run only as an audit record.
2. Extend remaining max-budget-not-converged deep runs beyond 50 epochs if the paper needs final convergence claims instead of reporting the current budget status.
3. Keep old `sample_id`-only group results only as appendix any-window diagnostics.
4. Rerun any experiment whose metrics show duplicate keys, missing score keys, or extra score keys.
5. Treat the score-grid best rows as validation-tuned candidates. The train-to-test multi-seed run above is the first holdout confirmation, but individual AUPRC is not strictly best because `individual_lof` is higher by `0.001061`; do not tune on the test split. Any further individual improvement must be selected on validation-only data, then confirmed on a fresh held-out run.
