# Dataset Extension Server Results 2026-05-25

This note records the server-side MOT-family dataset extension checkpoint.
It is a lightweight summary only: raw datasets, full score files, checkpoints,
logs, and credentials are not stored in the GitHub repository.

## Scope

Server run root:

```text
remote_runs/fusiontrack_dataset_extension_20260525
```

Code root used on the server:

```text
remote_runs/FusionTrack_repo
```

Datasets completed in this checkpoint:

| Dataset | Source used | Protocol |
| --- | --- | --- |
| MOT17 | `trackers` annotation mirror | `protocols/mot17_holdout_seed42` |
| SportsMOT | `trackers` annotation mirror | `protocols/sportsmot_holdout_seed42` |

M3OT is not claimed complete. The official Figshare private-link page and the
direct `ndownloader/files/52023875` URL both returned HTTP 403 from the server
on 2026-05-25, so the RGB/IR experiments still require external dataset access.

Additional access review on 2026-05-25:

- The Scientific Data/PMC article states that M3OT is freely available through
  `https://figshare.com/s/01fa8d1163f4e9a5a13a`, and that GT annotations use the
  MOTChallenge format while JSON annotations use the MS COCO format.
- The official GitHub repository `M3OT/M3OT` at commit
  `c56a75c60a03703756f96813c53b51d7ad353ca8` contains only `README.md` and
  tools such as `m3OT2coco.py` / `convert_M3OT_to_yolo.py`; it does not contain
  the dataset or an alternative downloader.
- Local `curl` access to both the Figshare private-link page and the direct
  `ndownloader/files/52023875?private_link=...` URL also returned HTTP 403.
  Therefore downloading locally and transferring to the server is not currently
  a viable workaround.
- Final server audit found only the official code/tools checkout at
  `remote_runs/dataset_sources/M3OT`; no RGB/IR image archive or MOT-style
  annotation data exists under `REMOTE_HOME` or `REMOTE_HOME/autodl-tmp`.

## Result Artifacts

Key server artifacts:

```text
combined_final_results.csv
combined_final_results.md
combined_best_by_owner.csv
stage_log.md
```

All selected official-source GPU baseline rows use clean validation JSONL for
convergence monitoring and the anomalous validation JSONL only for scoring via
`--score-jsonl`.

## Run Summary

| Run family | Purpose | Status |
| --- | --- | --- |
| `results/*/*/summary.csv` | Standard no-epoch/classical/FusionTrack matrix | Completed |
| `recent_official/` | 20 epoch GPU pilot | Completed, not converged |
| `recent_official_epoch80/` | First long-budget official pass | 3/16 converged |
| `recent_official_epoch200/` | Continued long-budget official pass | 10/16 selected tasks converged |
| `recent_official_epoch400/` | Continued hard-case pass | 4 additional selected tasks converged |
| `recent_official_epoch800/` | Final Timemixer individual pass | Final 2 selected tasks converged |

Final combined table:

```text
68 rows total
16 official-source GPU rows selected
16/16 selected official-source rows converged
68/68 rows passed strict key audit
```

## Selected Official-Source Budgets

| Dataset | Level | Method | Selected run | Final epoch | Best epoch | AUPRC | AUROC |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| MOT17 | individual | `official_catch` | epoch200 | 200 | 187 | 0.213346 | 0.598264 |
| MOT17 | individual | `official_cutaddpaste` | epoch80 | 80 | 75 | 0.134498 | 0.558824 |
| MOT17 | individual | `official_sensitive_hue` | epoch200 | 200 | 174 | 0.145217 | 0.619865 |
| MOT17 | individual | `official_timemixer` | epoch800 | 800 | 767 | 0.165184 | 0.592382 |
| MOT17 | group | `official_catch` | epoch200 | 200 | 180 | 0.008715 | 0.632292 |
| MOT17 | group | `official_cutaddpaste` | epoch200 | 200 | 168 | 0.011892 | 0.617612 |
| MOT17 | group | `official_sensitive_hue` | epoch200 | 200 | 195 | 0.007619 | 0.608007 |
| MOT17 | group | `official_timemixer` | epoch400 | 400 | 395 | 0.008696 | 0.665566 |
| SportsMOT | individual | `official_catch` | epoch200 | 200 | 190 | 0.237548 | 0.574599 |
| SportsMOT | individual | `official_cutaddpaste` | epoch80 | 80 | 75 | 0.246040 | 0.549519 |
| SportsMOT | individual | `official_sensitive_hue` | epoch400 | 400 | 373 | 0.174586 | 0.547489 |
| SportsMOT | individual | `official_timemixer` | epoch800 | 800 | 793 | 0.219579 | 0.563488 |
| SportsMOT | group | `official_catch` | epoch80 | 80 | 69 | 0.010359 | 0.539934 |
| SportsMOT | group | `official_cutaddpaste` | epoch200 | 200 | 117 | 0.011769 | 0.501052 |
| SportsMOT | group | `official_sensitive_hue` | epoch400 | 400 | 385 | 0.014202 | 0.563918 |
| SportsMOT | group | `official_timemixer` | epoch400 | 400 | 363 | 0.012531 | 0.552250 |

## Strongest Rows By AUPRC

| Dataset | Level | Strongest row | Owner | AUPRC | AUROC | Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| MOT17 | individual | `individual_lof` | classical baseline | 0.333332 | 0.718804 | Best FusionTrack row is `fusiontrack_individual_ensemble_tuned_topk`, AUPRC 0.267050, AUROC 0.740694. |
| MOT17 | group | `fusiontrack_group_hybrid_gated` | our method | 0.012278 | 0.688786 | `group_prediction_linear` is close: AUPRC 0.012110, AUROC 0.704211. |
| SportsMOT | individual | `individual_ocsvm` | classical baseline | 0.319550 | 0.719177 | Best FusionTrack row is `fusiontrack_individual_nn`, AUPRC 0.316217, AUROC 0.726255. |
| SportsMOT | group | `group_prediction_linear` | classical baseline | 0.018599 | 0.599109 | Best FusionTrack row is `fusiontrack_group_hybrid_gated`, AUPRC 0.015320, AUROC 0.604413. |

## Remaining Work

- M3OT still needs accessible RGB/IR data before experiments can run.
- The current M3OT blocker is external data access: a user-provided archive,
  institutional Figshare access, or a new public mirror is required before the
  same converter/protocol/benchmark pipeline can be executed.
- MOT17 and SportsMOT are completed under this holdout protocol, but final
  paper claims should still be checked with multi-seed or held-out test repeats
  if compute time permits.
- Large server artifacts should stay on the server or in external archives, not
  in the GitHub repository.
