# Recent Official-Source Baseline Candidates

This document records the 2024-2026 candidate baselines screened for the FusionTrack anomaly benchmark. A paper method can enter the main paper baseline table only when the result is produced from official or paper-linked source code and passes the strict key-audit protocol.

## Main-Table Candidates

| Method | Year | Venue/status | Official source | FusionTrack status |
| --- | ---: | --- | --- | --- |
| CATCH | 2025 | ICLR 2025 | https://github.com/decisionintelligence/CATCH | Added via `run_recent_official_fusiontrack.py --method catch`; external checkout only because no license file was found in the cloned repository. |
| CutAddPaste | 2024 | KDD 2024 | https://github.com/ruiking04/CutAddPaste | Added via `run_recent_official_fusiontrack.py --method cutaddpaste`; external checkout only because no license file was found in the cloned repository. |
| TimeMixer | 2024 | ICLR 2024 | https://github.com/kwuking/TimeMixer | Added via `run_recent_official_fusiontrack.py --method timemixer`; Apache-2.0 license found in the cloned repository. |

## Supplementary Official-Source Candidates

| Method | Year | Venue/status | Official source | FusionTrack status |
| --- | ---: | --- | --- | --- |
| SensitiveHUE | 2024/2025 | Public README says under review | https://github.com/yuesuoqingqiu/SensitiveHUE | Added as supplementary official-source candidate. Keep out of top-venue main-table claims unless a peer-reviewed venue/source record is verified. |
| DADA | 2025 | ICLR 2025 | https://github.com/iambowen/DADA | Candidate only. Official repository states that the pre-training code is not public, so only downstream/evaluation-style reproduction can be claimed. |
| UniTS | 2024 | NeurIPS 2024 | https://github.com/mims-harvard/UniTS | Candidate only. Official code is available under MIT; adapter requires a larger foundation-model workflow and checkpoint handling. |
| MOMENT | 2024 | ICML 2024 | https://github.com/moment-timeseries-foundation-model/moment | Candidate only. Official code is available under MIT; useful for foundation-model anomaly baselines but needs checkpoint/download and task adapter work. |
| D3R | 2023 | NeurIPS 2023 | https://github.com/ForestsKing/D3R | Candidate only. Official code is available, but diffusion training/inference cost is higher than the first recent-baseline pass. |

## Not Yet Main-Table Eligible

| Method | Reason |
| --- | --- |
| MtsCID | No author-official or paper-linked implementation was verified during this screening pass. Do not use the paper name in the main baseline table until official source is available. |
| Non-official reproductions of recent papers | These may be useful for reading, but they are not valid paper baselines under the project reproduction policy. |

## Current Remote Run

Remote result root:

```text
/root/autodl-tmp/fusiontrack_recent_official_20260522
```

The run uses RTX 5090 GPU execution, strict `sample_id` matching for individual rows, and strict `sample_id + window_id` matching for group rows.
